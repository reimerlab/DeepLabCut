"""
Function which are depricated in newer version of scipy (>1.2.1)
"""

from skimage.transform import resize

def imresize(img, scale_or_size, interp='bilinear'):
    """Drop-in replacement for scipy.misc.imresize"""
    if isinstance(scale_or_size, (int, float)):
        # Scale factor
        new_shape = (int(img.shape[0] * scale_or_size), 
                    int(img.shape[1] * scale_or_size))
    else:
        # Explicit size
        new_shape = scale_or_size
    
    # Map interpolation methods
    order_map = {
        'nearest': 0,
        'bilinear': 1, 
        'bicubic': 3
    }
    order = order_map.get(interp, 1)
    
    return resize(img, new_shape, 
                 order=order,
                 preserve_range=True, 
                 anti_aliasing=False).astype(img.dtype)
