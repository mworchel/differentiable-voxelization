import torch

def ARAP_energy(U: torch.Tensor,V: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
  """ Computes the ARAP energy for a given deformed mesh U (n x 3) with respect to the
  original mesh V (n x 3) and shared face indices F (m x 3).
  """
  device = U.device
  dtype = U.dtype

  V0 = V[F[:,0]]
  V1 = V[F[:,1]]
  V2 = V[F[:,2]]

  U0 = U[F[:,0]]
  U1 = U[F[:,1]]
  U2 = U[F[:,2]]

  VE0 = V2 - V1
  VE1 = V0 - V2
  VE2 = V1 - V0

  UE0 = U2 - U1
  UE1 = U0 - U2
  UE2 = U1 - U0

  cotanV0 = torch.sum(-VE1 * VE2, dim=1) / torch.linalg.vector_norm(torch.cross(-VE1, VE2, dim=1), dim=1)
  cotanV1 = torch.sum(-VE2 * VE0, dim=1) / torch.linalg.vector_norm(torch.cross(-VE2, VE0, dim=1), dim=1)
  cotanV2 = torch.sum(-VE0 * VE1, dim=1) / torch.linalg.vector_norm(torch.cross(-VE0, VE1, dim=1), dim=1)
  cotanV0.unsqueeze_(1), cotanV1.unsqueeze_(1), cotanV2.unsqueeze_(1)

  SV0 = torch.bmm(VE0.unsqueeze(2), (cotanV0 * UE0).unsqueeze(1))
  SV1 = torch.bmm(VE1.unsqueeze(2), (cotanV1 * UE1).unsqueeze(1))
  SV2 = torch.bmm(VE2.unsqueeze(2), (cotanV2 * UE2).unsqueeze(1))

  Fid0 = F[:,0].unsqueeze(1).unsqueeze(2).expand(-1, 3, 3)
  Fid1 = F[:,1].unsqueeze(1).unsqueeze(2).expand(-1, 3, 3)
  Fid2 = F[:,2].unsqueeze(1).unsqueeze(2).expand(-1, 3, 3)

  S = torch.zeros(V.size(dim=0), 3, 3, dtype=dtype, device=device)
  SF = SV0+SV1+SV2
  S.scatter_add_(0, Fid0, SF).scatter_add_(0, Fid1, SF).scatter_add_(0, Fid2, SF)

  X, sig, Y = torch.svd(S) # alternative: use the faster torch_batch_svd package 
  R = torch.bmm(X, Y.permute(0, 2, 1))
  # Need to flip the column of U corresponding to smallest singular value
  # for any det(Ri) <= 0
  entries_to_flip = torch.nonzero(torch.det(R) <= 0, as_tuple=False).flatten()  # idxs where det(R) <= 0
  if len(entries_to_flip) > 0:
    Xmod = X.clone()
    # cols_to_flip = torch.argmin(sig[entries_to_flip], dim=1)  # Get minimum singular value for each entry
    Xmod[entries_to_flip, :, -1] *= -1  # flip cols
    R[entries_to_flip] = torch.bmm(Xmod[entries_to_flip], Y[entries_to_flip].permute(0, 2, 1))

  R0 = R[F[:,0]]
  R1 = R[F[:,1]]
  R2 = R[F[:,2]]

  result = 0
  A = UE0 - torch.bmm(VE0.unsqueeze(1), R0).squeeze()
  result += torch.sum(cotanV0 * A*A)
  A = UE1 - torch.bmm(VE1.unsqueeze(1), R0).squeeze()
  result += torch.sum(cotanV1 * A*A)
  A = UE2 - torch.bmm(VE2.unsqueeze(1), R0).squeeze()
  result += torch.sum(cotanV2 * A*A)

  A = UE0 - torch.bmm(VE0.unsqueeze(1), R1).squeeze()
  result += torch.sum(cotanV0 * A*A)
  A = UE1 - torch.bmm(VE1.unsqueeze(1), R1).squeeze()
  result += torch.sum(cotanV1 * A*A)
  A = UE2 - torch.bmm(VE2.unsqueeze(1), R1).squeeze()
  result += torch.sum(cotanV2 * A*A)

  A = UE0 - torch.bmm(VE0.unsqueeze(1), R2).squeeze()
  result += torch.sum(cotanV0 * A*A)
  A = UE1 - torch.bmm(VE1.unsqueeze(1), R2).squeeze()
  result += torch.sum(cotanV1 * A*A)
  A = UE2 - torch.bmm(VE2.unsqueeze(1), R2).squeeze()
  result += torch.sum(cotanV2 * A*A)

  return result
