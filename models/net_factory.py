from models import feat3dnet

networks_map = {'3DFeatNet': feat3dnet.Feat3dNet,
                }


def get_network(name):

    model = networks_map[name]
    return model
