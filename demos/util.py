import torch

def load_obj(path: str, texture_coords = False, swap_yz = False, device: torch.device = 'cpu') -> list[torch.Tensor]:
    """
    Loads a triangle or quad mesh from an .obj file.

    Args:
      path (str): Path to the .obj file.
      texture_coords (bool): Whether to also load texture coordinates. Defaults to False.
      swap_yz (bool): If True, makes y the new z axis. Defaults to False.
      device (torch.device): Device for the output tensors. Defaults to 'cpu'.

    Returns:
      tuple: A tuple containing:
        - torch.Tensor: Vertex tensor of shape (#V, 3).
        - list[list] or torch.Tensor: Face connectivity of shape (#F, d).
          Returned as a tensor if d is the same for all faces, otherwise a list of lists.
        - torch.Tensor: (Only if texture_coords=True) Vertex texture coordinates of shape (#V, 2).
        - list[list]: (Only if texture_coords=True) Vertex texture indices of shape (#F, 3|4).
    """
    with open(path, 'r') as file:
      lines = file.readlines()

    vertices         = []
    texcoords        = []
    faces            = []
    texcoordids      = []

    equal_face_n = None
    for idx, line in enumerate(lines):
      if line.startswith("v "):
        vertices.append([float(v) for v in line.split(" ")[1:]])
      if texture_coords and line.startswith("vt "):
        texcoords.append([float(vt) for vt in line.split(" ")[1:]])
      if line.startswith("f "):
        face_strings = line.split(" ")[1:]
        vis = [int(fv.split("/")[0]) - 1 for fv in face_strings]
        faces.append(vis)

        if texture_coords:
          vtis = [int(fv.split("/")[1]) - 1 for fv in face_strings]
          texcoordids.append(vtis)

    n = len(faces[0]) if len(faces) > 0 else 0
    if all([len(f) == n for f in faces]):
      faces = torch.tensor(faces)

    v  = torch.tensor(vertices, dtype=torch.float32, device=device)
    if swap_yz:
      tmp_y = v[:,1].clone()
      v[:,1] = -v[:,2]
      v[:,2] = tmp_y
    
    if not texture_coords:
      return v, faces
    vt = torch.tensor(texcoords, dtype=torch.float32, device=device)                
    return v, faces, vt, texcoordids


def rotation3D(angles: torch.Tensor) -> torch.Tensor:
    """Generate a rotation matrix from yaw, pitch and roll angles.

    Args:
      angles (torch.Tensor): Angle tensor with batches in radian (yaw, pitch, roll)
        of shape (b, 3).

    Returns:
      torch.Tensor: Rotation matrix of shape (b, 3, 3).
    """
    yaw, pitch, roll = angles.T
    c_a = torch.cos(yaw)
    s_a = torch.sin(yaw)
    c_b = torch.cos(pitch)
    s_b = torch.sin(pitch)
    c_c = torch.cos(roll)
    s_c = torch.sin(roll)

    R = torch.stack([
        c_a*c_b, c_a*s_b*s_c-s_a*c_c, c_a*s_b*c_c+s_a*s_c,
        s_a*c_b, s_a*s_b*s_c+c_a*c_c, s_a*s_b*c_c-c_a*s_c,
        -s_b,    c_b*s_c,             c_b*c_c
    ], dim=1)
    return R.view(-1, 3, 3)
