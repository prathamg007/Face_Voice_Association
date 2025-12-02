import torch
import torch.nn.functional as F

def fop_loss(a_vec, v_vec, fused_vec):
    """
    Fusion and Orthogonal Projection (FOP) Loss.
    Encourages modality-specific vectors to align with the fused vector representation.
    """
    a_norm = F.normalize(a_vec, p=2, dim=1)
    v_norm = F.normalize(v_vec, p=2, dim=1)
    f_norm = F.normalize(fused_vec, p=2, dim=1)
    
    # Minimize distance (maximize cosine sim) between modalities and fused vector
    loss_a = (f_norm * a_norm).sum(1).abs().mean()
    loss_v = (f_norm * v_norm).sum(1).abs().mean()
    
    return loss_a + loss_v

def nt_xent_loss(a_vec, v_vec, temperature=0.07):
    """
    Normalized Temperature-scaled Cross Entropy Loss (Contrastive).
    Assumes a_vec and v_vec are positive pairs at the same index in the batch.
    """
    if a_vec.shape[0] < 2 or v_vec.shape[0] < 2:
        return torch.tensor(0.0, device=a_vec.device)
        
    a_norm = F.normalize(a_vec, dim=1)
    v_norm = F.normalize(v_vec, dim=1)
    
    # Cosine similarity matrix
    sim = torch.matmul(a_norm, v_norm.T) / temperature
    
    # Positives are on the diagonal
    pos = torch.exp(torch.diag(sim))
    exp_sim = torch.exp(sim)
    
    # Sum of exponentials for denominator
    denom = exp_sim.sum(dim=1)
    
    loss = -torch.log(pos / (denom + 1e-12))
    return loss.mean()