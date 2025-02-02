import torch
from math import pi


def deg2rad(angles):
    return angles * pi / 180.


def rad2deg(angles):
    return angles * 180. / pi


def rotation_matrix_y(angle):
    """Returns rotation matrix about y-axis.

    Args:
        angle (torch.Tensor): Rotation angle in degrees. Shape (batch_size,).
    """
    # Initialize rotation matrix
    rotation_matrix = torch.zeros(angle.shape[0], 3, 3, device=angle.device)
    # Fill out matrix entries
    angle_rad = deg2rad(angle)
    cos_angle = torch.cos(angle_rad)
    sin_angle = torch.sin(angle_rad)
    rotation_matrix[:, 0, 0] = cos_angle
    rotation_matrix[:, 0, 2] = sin_angle
    rotation_matrix[:, 1, 1] = 1.
    rotation_matrix[:, 2, 0] = -sin_angle
    rotation_matrix[:, 2, 2] = cos_angle
    return rotation_matrix


def rotation_matrix_z(angle):
    """Returns rotation matrix about z-axis.

    Args:
        angle (torch.Tensor): Rotation angle in degrees. Shape (batch_size,).
    """
    # Initialize rotation matrix
    rotation_matrix = torch.zeros(angle.shape[0], 3, 3, device=angle.device)
    # Fill out matrix entries
    angle_rad = deg2rad(angle)
    cos_angle = torch.cos(angle_rad)
    sin_angle = torch.sin(angle_rad)
    rotation_matrix[:, 0, 0] = cos_angle
    rotation_matrix[:, 0, 1] = -sin_angle
    rotation_matrix[:, 1, 0] = sin_angle
    rotation_matrix[:, 1, 1] = cos_angle
    rotation_matrix[:, 2, 2] = 1.
    return rotation_matrix


def translate(matrix, translations):
    matrix2 = torch.zeros(matrix.shape[0], 4, 4, device=matrix.device)
    matrix2[:, 0:3, 0:3] = matrix
    matrix2[:, 3, 3] = 1.
    matrix2[:, :3, 3] = translations
    return matrix2


def azimuth_elevation_to_rotation_matrix(azimuth, elevation, translations):
    """Returns rotation matrix matching the default view (i.e. both azimuth and
    elevation are zero) to the view defined by the azimuth, elevation pair.


    Args:
        azimuth (torch.Tensor): Shape (batch_size,). Azimuth of camera in
            degrees.
        elevation (torch.Tensor): Shape (batch_size,). Elevation of camera in
            degrees.

    Notes:
        The azimuth and elevation refer to the position of the camera. This
        function returns the rotation of the *scene representation*, i.e. the
        inverse of the camera transformation.
    """
    # In the coordinate system we define (see README), azimuth rotation
    # corresponds to negative rotation about y axis and elevation rotation to a
    # negative rotation about z axis
    # print(azimuth, elevation, translations)
    azimuth_matrix = rotation_matrix_y(-azimuth)
    elevation_matrix = rotation_matrix_z(-elevation)

    # We first perform elevation rotation followed by azimuth when rotating camera
    camera_matrix = azimuth_matrix @ elevation_matrix
    # print('camera matrix', camera_matrix, camera_matrix.shape)
    translated_camera_matrix = translate(camera_matrix, translations)
    
    # Object rotation matrix is inverse (i.e. transpose) of camera rotation matrix
    transpose_camera_matrix = transpose_matrix(translated_camera_matrix)
    # print('transpose_camera_matrix', transpose_camera_matrix, transpose_camera_matrix.shape)
    return transpose_camera_matrix


def rotation_matrix_source_to_target(azimuth_source, elevation_source, translations_source,
                                     azimuth_target, elevation_target, translations_target):
    """Returns rotation matrix matching two views defined by azimuth, elevation
    pairs.

    Args:
        azimuth_source (torch.Tensor): Shape (batch_size,). Azimuth of source
            view in degrees.
        elevation_source (torch.Tensor): Shape (batch_size,). Elevation of
            source view in degrees.
        azimuth_target (torch.Tensor): Shape (batch_size,). Azimuth of target
            view in degrees.
        elevation_target (torch.Tensor): Shape (batch_size,). Elevation of
            target view in degrees.
    """
    # Calculate rotation matrix for each view
    rotation_source = azimuth_elevation_to_rotation_matrix(azimuth_source,
                                                           elevation_source,
                                                           torch.zeros_like(translations_source))
    rotation_target = azimuth_elevation_to_rotation_matrix(azimuth_target,
                                                           elevation_target,
                                                           translations_target - translations_source)
    # Calculate rotation matrix bringing source view to target view (note that
    # for rotation matrix, inverse is transpose)

    # print('rotation_source', rotation_source, rotation_source.shape)
    # print('rotation_target', rotation_target, rotation_target.shape)
    # print('returning', rotation_target @ transpose_matrix(rotation_source))
    # print('translations source', translations_source)
    # print('translations target', translations_target)

    return rotation_target @ transpose_matrix(rotation_source)


def transpose_matrix(matrix):
    """Transposes a batch of matrices.

    Args:
        matrix (torch.Tensor): Batch of matrices of shape (batch_size, n, m).
    """
    return matrix.transpose(1, 2)
