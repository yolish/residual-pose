import torch
import torch.nn as nn
import torch.nn.functional as F
from models.residuals import select_centroids, add_residuals

class ResidualPoseNet(nn.Module):
    """
    A class to represent a classic pose regressor (PoseNet) with an efficient-net backbone with/without residuals
    """
    def __init__(self, config):
        """
        Constructor
        :param backbone_path: backbone path
        """
        super(ResidualPoseNet, self).__init__()

        backbone_path = config.get("backbone_path") # change to something of-the-shelf
        self.backbone = torch.load(backbone_path)
        backbone_dim = config.get("backbone_dim") # 1280, 2048...
        position_num_classes = config.get("position_num_classes")
        orientation_num_classes = config.get("orientation_num_classes")


        self.with_residuals = False
        if position_num_classes is not None:
            self.with_residuals = True

        latent_dim = config.get("latent_dim") # 1024

        # Regressor layers
        self.fc_latent = nn.Linear(backbone_dim, latent_dim)
        self.position_reg = nn.Linear(latent_dim, 3)
        self.orientation_reg = nn.Linear(latent_dim, 4)
        if self.with_residuals:
            self.fc_latent_position = nn.Linear(latent_dim, latent_dim)
            self.fc_latent_orientation = nn.Linear(latent_dim, latent_dim)
            self.position_cls_embed = nn.Linear(latent_dim, position_num_classes)
            self.orientation_cls_embed = nn.Linear(latent_dim, orientation_num_classes)
            self.log_softmax = nn.LogSoftmax(dim=1)
            '''
            self.position_centroid_encode = nn.Linear(3, latent_dim)
            self.orientation_centroid_encode = nn.Linear(4, latent_dim)
            '''

        self.dropout = nn.Dropout(p=config.get("dropout"))
        self.avg_pooling_2d = nn.AdaptiveAvgPool2d(1)

        # Initialize FC layers
        for m in list(self.modules()):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, data, centroids=None, cls_indices=None):
        """
        Forward pass
        :param data: (torch.Tensor) dictionary with key-value 'img' -- input image (N X C X H X W)
        :return: (torch.Tensor) dictionary with key-value 'pose' -- 7-dimensional absolute pose for (N X 7)
        """
        x = self.backbone.extract_features(data.get('img')) # change to usual forwards pass
        x = self.avg_pooling_2d(x)
        out = x.flatten(start_dim=1)
        x = F.relu(self.fc_latent(out))

        output = {}
        if self.with_residuals:
            position_centroids, orientation_centroids = centroids
            gt_position_cls = None
            gt_orientation_cls = None
            if cls_indices is not None:
                gt_position_cls, gt_orientation_cls = cls_indices

            position_latent = self.fc_latent_position(x)
            position_cls_log_distr = self.log_softmax(
                self.position_cls_embed(position_latent))
            output["position_cls"] = position_cls_log_distr
            position_residuals = self.position_reg(position_latent)
            p_x = add_residuals(position_cls_log_distr, position_centroids,
                                position_residuals, gt_indices=gt_position_cls)

            orientation_latent = self.fc_latent_orientation(x)
            orientation_cls_log_distr = self.log_softmax(
                self.position_cls_embed(orientation_latent))
            output["orientation_cls"] = orientation_cls_log_distr
            orientation_residuals = self.orientation_reg(orientation_latent)
            p_q = add_residuals(orientation_cls_log_distr, orientation_centroids,
                                orientation_residuals, gt_indices=gt_orientation_cls)

            '''
            position_latent = self.fc_latent_position(x)
            position_cls_log_distr = self.log_softmax(
                self.position_cls_embed(x))
            output["position_cls"] = position_cls_log_distr
            position_centroids = select_centroids(position_cls_log_distr, position_centroids)
            latent_position = position_latent + self.position_centroid_encode(position_centroids)
            p_x = self.position_reg(latent_position)

            orientation_latent = self.fc_latent_orientation(x)
            orientation_cls_log_distr = self.log_softmax(
                self.orientation_cls_embed(x))
            output["orientation_cls"] = orientation_cls_log_distr
            orientation_centroids = select_centroids(orientation_cls_log_distr, orientation_centroids)
            latent_orientation = orientation_latent + self.orientation_centroid_encode(orientation_centroids)
            p_q = self.orientation_reg(latent_orientation)
            '''



            #p_x = self.dropout(F.relu(self.fc_latent(out + self.position_centroid_encode(position_centroids))))
            #p_q = self.dropout(F.relu(self.fc_latent(out + self.orientation_centroid_encode(orientation_centroids))))
            #p_x = self.position_reg(p_x) # + position_centroids
            #p_q = self.orientation_reg(p_q) # + orientation_centroids



        else:
            x = self.dropout(x)
            p_x = self.position_reg(x)
            p_q = self.orientation_reg(x)

        output['pose'] = torch.cat((p_x, p_q), dim=1)
        return output

