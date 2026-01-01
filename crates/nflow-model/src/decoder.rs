use crate::encoder::GatedConvBlock;
use anyhow::Result;
use candle_core::{Device, Tensor, D};
use candle_nn::{Linear, Module, VarBuilder};

pub struct SinusoidalTimestepEmbedding {
    dim: usize,
    freqs: Tensor,
}

impl SinusoidalTimestepEmbedding {
    pub fn new(dim: usize, max_period: f32, device: &Device) -> Result<Self> {
        let half = dim / 2;
        let freqs: Vec<f32> = (0..half)
            .map(|i| (-((i as f32) / (half as f32)) * max_period.ln()).exp())
            .collect();
        let freqs = Tensor::from_vec(freqs, (1, half), device)?;
        Ok(Self { dim, freqs })
    }

    pub fn forward(&self, t: &Tensor) -> Result<Tensor> {
        let t = t.reshape(((), 1))?; 
        let args = t.broadcast_mul(&self.freqs)?; 
        let sin = args.sin()?;
        let cos = args.cos()?;
        let out = Tensor::cat(&[&sin, &cos], 1)?; 
        
        if self.dim % 2 == 1 {
            let zeros = Tensor::zeros((out.dim(0)?, 1), out.dtype(), out.device())?;
            Ok(Tensor::cat(&[&out, &zeros], 1)?)
        } else {
            Ok(out)
        }
    }
}

pub struct FlowMatchingDecoder {
    input_proj: Linear,
    time_proj: Linear,
    blocks: Vec<GatedConvBlock>,
    output_proj: Linear,
    time_embedding: SinusoidalTimestepEmbedding,
}

impl FlowMatchingDecoder {
    pub fn new(
        hidden_dim: usize, 
        output_dim: usize, 
        enc_output_dim: usize,
        num_layers: usize, 
        vs: VarBuilder
    ) -> Result<Self> {
        // x_t (output_dim) + cond (enc_output_dim) + t_emb (hidden_dim)
        let input_dim = output_dim + enc_output_dim + hidden_dim;
        
        let input_proj = candle_nn::linear(input_dim, hidden_dim, vs.pp("input_proj"))?;
        let time_proj = candle_nn::linear(hidden_dim, hidden_dim, vs.pp("time_proj"))?;
        let output_proj = candle_nn::linear(hidden_dim, output_dim, vs.pp("output_proj"))?;

        let time_embedding = SinusoidalTimestepEmbedding::new(hidden_dim, 10000.0, vs.device())?;

        let mut blocks = Vec::with_capacity(num_layers);
        let blocks_vs = vs.pp("blocks");
        for i in 0..num_layers {
            blocks.push(GatedConvBlock::new(hidden_dim, 5, blocks_vs.pp(i))?);
        }

        Ok(Self { 
            input_proj, 
            time_proj, 
            blocks, 
            output_proj,
            time_embedding,
        })
    }

    pub fn compute_vector_field(&self, x: &Tensor, t: &Tensor, cond: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;
        
        let t_emb = self.time_embedding.forward(t)?;
        let t_emb = self.time_proj.forward(&t_emb)?; 
        
        // 時間埋め込みを系列方向に拡張
        let t_emb = t_emb.unsqueeze(1)?.broadcast_as((b_sz, seq_len, t_emb.dim(1)?))?;

        // 入力を結合: [x(現在のノイズ), cond(エンコーダ出力等), t_emb(時間)]
        let inputs = Tensor::cat(&[x, cond, &t_emb], D::Minus1)?.contiguous()?;
        let mut h = self.input_proj.forward(&inputs)?;

        for block in &self.blocks {
            h = block.forward(&h)?;
        }

        let v = self.output_proj.forward(&h)?;
        Ok(v)
    }

    pub fn sample(&self, cond: &Tensor, steps: usize, device: &Device) -> Result<Tensor> {
        let batch_size = cond.dim(0)?;
        let seq_len = cond.dim(1)?;
        let out_dim = self.output_proj.weight().dim(0)?;
        
        let mut x = Tensor::randn(0f32, 1f32, (batch_size, seq_len, out_dim), device)?;
        let dt = 1.0 / steps as f32;

        for i in 0..steps {
            let t_val = i as f32 * dt;
            let t = Tensor::from_slice(&[t_val], (1,), device)?;
            
            // Midpoint法 (RK2)
            // 1. 現在地点での勾配 v1
            let v1 = self.compute_vector_field(&x, &t, cond)?;
            
            // 2. 中間地点での勾配 v2
            let t_mid = Tensor::from_slice(&[t_val + 0.5 * dt], (1,), device)?;
            let x_mid = x.add(&(v1.affine(0.5 * dt as f64, 0.0)?))?;
            let v2 = self.compute_vector_field(&x_mid, &t_mid, cond)?;
            
            // 3. 中間地点の勾配で更新
            x = x.add(&(v2.affine(dt as f64, 0.0)?))?;
        }

        Ok(x)
    }
}
