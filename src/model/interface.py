from model.model import aot_gan, deepfill, stable_diffusion

def get_model(name):
    match name:
        case 'stable_diffussion':
            return stable_diffusion()
        case 'aot_gan':
            return aot_gan()
        case 'deepfill':
            return deepfill()
        case _:
            raise ValueError(f'Unknown model name: {name}')
