pub mod encoder;
pub mod decoder;

use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;

pub use encoder::Encoder;
pub use decoder::FlowMatchingDecoder;

pub struct NanoFlowModel {
    encoder: Encoder,
    decoder_coarse: FlowMatchingDecoder, // 低次メル (16次元)
    decoder_fine: FlowMatchingDecoder,   // 高次メル (64次元)
}

impl NanoFlowModel {
    pub fn new(
        vocab_size: usize,
        enc_dim: usize,
        enc_layers: usize,
        _dec_output_dim: usize, // 固定で 80 (16 + 64) を扱う
        dec_layers: usize,
        vs: VarBuilder,
    ) -> Result<Self> {
        let encoder = Encoder::new(vocab_size, enc_dim, enc_layers, vs.pp("encoder"))?;
        
        // Coarse: 音素コード(enc_dim) -> メル(16次元)
        let decoder_coarse = FlowMatchingDecoder::new(
            enc_dim, 
            16, 
            enc_dim, 
            dec_layers, 
            vs.pp("decoder_coarse")
        )?;

        // Fine: 音素コード(enc_dim) + Coarseメル(16次元) -> 残りメル(64次元)
        let decoder_fine = FlowMatchingDecoder::new(
            enc_dim, 
            64, 
            enc_dim + 16, 
            dec_layers, 
            vs.pp("decoder_fine")
        )?;

        Ok(Self { encoder, decoder_coarse, decoder_fine })
    }

    /// 学習時: CoarseとFineの両方のベクトル場を計算
    pub fn forward_train(
        &self,
        phonemes: &Tensor,
        accents: &Tensor,
        x_t: &Tensor, // [batch, seq, 80]
        t: &Tensor,   // [batch]
        clean_mel: &Tensor, // Teacher Forcing 用の正解 [batch, seq, 80]
    ) -> Result<(Tensor, Tensor)> {
        let cond = self.encoder.forward(phonemes, accents)?;
        
        // 16次元と64次元に分割
        let x_t_coarse = x_t.narrow(2, 0, 16)?.contiguous()?;
        let x_t_fine = x_t.narrow(2, 16, 64)?.contiguous()?;
        
        // Coarse のベクトル場予測
        let v_coarse = self.decoder_coarse.compute_vector_field(&x_t_coarse, t, &cond)?;
        
        // Fine の条件: 音素コード + 正解の低次メル (Teacher Forcing)
        let clean_coarse = clean_mel.narrow(2, 0, 16)?.contiguous()?;
        let cond_fine = Tensor::cat(&[&cond, &clean_coarse], 2)?;
        let v_fine = self.decoder_fine.compute_vector_field(&x_t_fine, t, &cond_fine)?;
        
        Ok((v_coarse, v_fine))
    }

    /// 推論時: 階層的に生成
    pub fn generate(
        &self,
        phonemes: &Tensor,
        accents: &Tensor,
        steps: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let cond = self.encoder.forward(phonemes, accents)?;
        
        // Step 1: Coarse 生成
        let x_coarse = self.decoder_coarse.sample(&cond, steps, device)?;
        
        // Step 2: Fine 生成 (Coarse を追加条件にする)
        let cond_fine = Tensor::cat(&[&cond, &x_coarse], 2)?;
        let x_fine = self.decoder_fine.sample(&cond_fine, steps, device)?;
        
        // 合体させて 80次元にする
        let x = Tensor::cat(&[x_coarse, x_fine], 2)?;
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Module};

    #[test]
    fn test_model_flow() -> Result<()> {
        let device = Device::Cpu;
        let vs = VarBuilder::zeros(DType::F32, &device);
        let model = NanoFlowModel::new(100, 64, 2, 80, 2, vs)?;

        let b_sz = 2;
        let seq_len = 10;
        let phonemes = Tensor::zeros((b_sz, seq_len), DType::U32, &device)?;
        let accents = Tensor::zeros((b_sz, seq_len), DType::U32, &device)?;

        // 推論テスト
        let output = model.generate(&phonemes, &accents, 5, &device)?;
        assert_eq!(output.dims(), &[b_sz, seq_len, 80]);

        Ok(())
    }
}
