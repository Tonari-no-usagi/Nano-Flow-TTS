use anyhow::Result;
use candle_core::{Tensor, D};
use candle_nn::{Embedding, LayerNorm, Linear, Module, VarBuilder, Conv1d, Conv1dConfig};

/// Gated Linear Unit (GLU)
/// ゲート機構により情報の流れを制御する軽量な活性化関数
struct Glu {
    linear: Linear,
}

impl Glu {
    fn new(dim: usize, vs: VarBuilder) -> Result<Self> {
        // 出力は dim * 2 で、半分をゲートにする
        let linear = candle_nn::linear(dim, dim * 2, vs)?;
        Ok(Self { linear })
    }
}

impl Module for Glu {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let xs = self.linear.forward(xs)?;
        let chunks = xs.chunk(2, D::Minus1)?;
        let (val, gate) = (&chunks[0], &chunks[1]);
        val * candle_nn::ops::sigmoid(gate)?
    }
}

/// 1D畳み込みを使用した軽量ブロック
pub struct GatedConvBlock {
    conv: Conv1d,
    glu: Glu,
    norm: LayerNorm,
}

impl GatedConvBlock {
    pub fn new(dim: usize, kernel_size: usize, vs: VarBuilder) -> Result<Self> {
        let cfg = Conv1dConfig {
            padding: kernel_size / 2,
            ..Default::default()
        };
        // [batch, dim, seq] を想定
        let conv = candle_nn::conv1d(dim, dim, kernel_size, cfg, vs.pp("conv"))?;
        let glu = Glu::new(dim, vs.pp("glu"))?;
        let norm = candle_nn::layer_norm(dim, 1e-5, vs.pp("norm"))?;
        Ok(Self { conv, glu, norm })
    }
}

impl Module for GatedConvBlock {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let residual = xs;
        // xs: [batch, seq, dim] -> [batch, dim, seq]
        let mut h = xs.transpose(1, 2)?;
        h = self.conv.forward(&h)?;
        // [batch, dim, seq] -> [batch, seq, dim]
        h = h.transpose(1, 2)?;
        h = self.glu.forward(&h)?;
        h = self.norm.forward(&h)?;
        h + residual
    }
}

pub struct Encoder {
    phoneme_emb: Embedding,
    accent_emb: Embedding,
    blocks: Vec<GatedConvBlock>,
}

impl Encoder {
    pub fn new(
        vocab_size: usize,
        dim: usize,
        num_layers: usize,
        vs: VarBuilder,
    ) -> Result<Self> {
        let phoneme_emb = candle_nn::embedding(vocab_size, dim, vs.pp("phoneme_emb"))?;
        let accent_emb = candle_nn::embedding(2, dim, vs.pp("accent_emb"))?;
        let mut blocks = Vec::with_capacity(num_layers);
        let blocks_vs = vs.pp("blocks");
        for i in 0..num_layers {
            // 音声の局所相関に合わせ、カーネルサイズは 5 程度にする
            blocks.push(GatedConvBlock::new(dim, 5, blocks_vs.pp(i))?);
        }
        Ok(Self {
            phoneme_emb,
            accent_emb,
            blocks,
        })
    }

    pub fn forward(&self, phonemes: &Tensor, accents: &Tensor) -> Result<Tensor> {
        let p_emb = self.phoneme_emb.forward(phonemes)?;
        let a_emb = self.accent_emb.forward(accents)?;
        let mut xs = (p_emb + a_emb)?;

        for block in &self.blocks {
            xs = block.forward(&xs)?;
        }
        Ok(xs)
    }
}
