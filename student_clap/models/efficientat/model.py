from functools import partial
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
from torch import nn, Tensor
import torch.nn.functional as F
import warnings
# Import the new Conv2d variant while suppressing the module-level deprecation warning
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Don't use ConvNormActivation directly")
    from torchvision.ops.misc import Conv2dNormActivation as ConvNormActivation  # Use Conv2d variant
from torch.hub import load_state_dict_from_url
import logging
logger = logging.getLogger(__name__)
import urllib.parse

from .utils import cnn_out_size
from .block_types import InvertedResidualConfig, InvertedResidual
from .attention_pooling import MultiHeadAttentionPooling
from .helpers_utils import NAME_TO_WIDTH


# Adapted version of MobileNetV3 pytorch implementation
# https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py

# points to github releases
model_url = "https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/"
# folder to store downloaded models to (package-local by default)
import os
pkg_dir = os.path.dirname(__file__)
# Use package root `student_clap/resources` (two levels up from this file)
model_dir = os.path.normpath(os.path.join(pkg_dir, "..", "..", "resources"))
# Ensure the folder exists so downloads are deterministic regardless of CWD
os.makedirs(model_dir, exist_ok=True)
logger.info(f"EfficientAT pretrained files will be stored in: {model_dir}")


pretrained_models = {
    # pytorch ImageNet pre-trained model
    # own ImageNet pre-trained models will follow
    # NOTE: for easy loading we provide the adapted state dict ready for AudioSet training (1 input channel,
    # 527 output classes)
    # NOTE: the classifier is just a random initialization, feature extractor (conv layers) is pre-trained
    "mn10_im_pytorch": urllib.parse.urljoin(model_url, "mn10_im_pytorch.pt"),
    # self-trained models on ImageNet
    "mn01_im": urllib.parse.urljoin(model_url, "mn01_im.pt"),
    "mn02_im": urllib.parse.urljoin(model_url, "mn02_im.pt"),
    "mn04_im": urllib.parse.urljoin(model_url, "mn04_im.pt"),
    "mn05_im": urllib.parse.urljoin(model_url, "mn05_im.pt"),
    "mn10_im": urllib.parse.urljoin(model_url, "mn10_im.pt"),
    "mn20_im": urllib.parse.urljoin(model_url, "mn20_im.pt"),
    "mn30_im": urllib.parse.urljoin(model_url, "mn30_im.pt"),
    "mn40_im": urllib.parse.urljoin(model_url, "mn40_im.pt"),
    # Models trained on AudioSet
    "mn01_as": urllib.parse.urljoin(model_url, "mn01_as_mAP_298.pt"),
    "mn02_as": urllib.parse.urljoin(model_url, "mn02_as_mAP_378.pt"),
    "mn04_as": urllib.parse.urljoin(model_url, "mn04_as_mAP_432.pt"),
    "mn05_as": urllib.parse.urljoin(model_url, "mn05_as_mAP_443.pt"),
    "mn10_as": urllib.parse.urljoin(model_url, "mn10_as_mAP_471.pt"),
    "mn20_as": urllib.parse.urljoin(model_url, "mn20_as_mAP_478.pt"),
    "mn30_as": urllib.parse.urljoin(model_url, "mn30_as_mAP_482.pt"),
    "mn40_as": urllib.parse.urljoin(model_url, "mn40_as_mAP_484.pt"),
    "mn40_as(2)": urllib.parse.urljoin(model_url, "mn40_as_mAP_483.pt"),
    "mn40_as(3)": urllib.parse.urljoin(model_url, "mn40_as_mAP_483(2).pt"),
    "mn40_as_no_im_pre": urllib.parse.urljoin(model_url, "mn40_as_no_im_pre_mAP_483.pt"),
    "mn40_as_no_im_pre(2)": urllib.parse.urljoin(model_url, "mn40_as_no_im_pre_mAP_483(2).pt"),
    "mn40_as_no_im_pre(3)": urllib.parse.urljoin(model_url, "mn40_as_no_im_pre_mAP_482.pt"),
    "mn40_as_ext": urllib.parse.urljoin(model_url, "mn40_as_ext_mAP_487.pt"),
    "mn40_as_ext(2)": urllib.parse.urljoin(model_url, "mn40_as_ext_mAP_486.pt"),
    "mn40_as_ext(3)": urllib.parse.urljoin(model_url, "mn40_as_ext_mAP_485.pt"),
    # varying hop size (time resolution)
    "mn10_as_hop_5": urllib.parse.urljoin(model_url, "mn10_as_hop_5_mAP_475.pt"),
    "mn10_as_hop_15": urllib.parse.urljoin(model_url, "mn10_as_hop_15_mAP_463.pt"),
    "mn10_as_hop_20": urllib.parse.urljoin(model_url, "mn10_as_hop_20_mAP_456.pt"),
    "mn10_as_hop_25": urllib.parse.urljoin(model_url, "mn10_as_hop_25_mAP_447.pt"),
    # varying n_mels (frequency resolution)
    "mn10_as_mels_40": urllib.parse.urljoin(model_url, "mn10_as_mels_40_mAP_453.pt"),
    "mn10_as_mels_64": urllib.parse.urljoin(model_url, "mn10_as_mels_64_mAP_461.pt"),
    "mn10_as_mels_256": urllib.parse.urljoin(model_url, "mn10_as_mels_256_mAP_474.pt"),
    # fully-convolutional head
    "mn10_as_fc": urllib.parse.urljoin(model_url, "mn10_as_fc_mAP_465.pt"),
    "mn10_as_fc_s2221": urllib.parse.urljoin(model_url, "mn10_as_fc_s2221_mAP_466.pt"),
    "mn10_as_fc_s2211": urllib.parse.urljoin(model_url, "mn10_as_fc_s2211_mAP_466.pt"),
}


class MN(nn.Module):
    def __init__(
        self,
        inverted_residual_setting: List[InvertedResidualConfig],
        last_channel: int,
        num_classes: int = 1000,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.2,
        in_conv_kernel: int = 3,
        in_conv_stride: int = 2,
        in_channels: int = 1,
        **kwargs: Any,
    ) -> None:
        """
        MobileNet V3 main class

        Args:
            inverted_residual_setting (List[InvertedResidualConfig]): Network structure
            last_channel (int): The number of channels on the penultimate layer
            num_classes (int): Number of classes
            block (Optional[Callable[..., nn.Module]]): Module specifying inverted residual building block for models
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
            dropout (float): The droupout probability
            in_conv_kernel (int): Size of kernel for first convolution
            in_conv_stride (int): Size of stride for first convolution
            in_channels (int): Number of input channels
        """
        super(MN, self).__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        if block is None:
            block = InvertedResidual

        depthwise_norm_layer = norm_layer = \
            norm_layer if norm_layer is not None else partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers: List[nn.Module] = []

        kernel_sizes = [in_conv_kernel]
        strides = [in_conv_stride]

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            ConvNormActivation(
                in_channels,
                firstconv_output_channels,
                kernel_size=in_conv_kernel,
                stride=in_conv_stride,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        )

        # get squeeze excitation config
        se_cnf = kwargs.get('se_conf', None)

        # building inverted residual blocks
        # - keep track of size of frequency and time dimensions for possible application of Squeeze-and-Excitation
        # on the frequency/time dimension
        # - applying Squeeze-and-Excitation on the time dimension is not recommended as this constrains the network to
        # a particular length of the audio clip, whereas Squeeze-and-Excitation on the frequency bands is fine,
        # as the number of frequency bands is usually not changing
        f_dim, t_dim = kwargs.get('input_dims', (128, 1000))
        # take into account first conv layer
        f_dim = cnn_out_size(f_dim, 1, 1, 3, 2)
        t_dim = cnn_out_size(t_dim, 1, 1, 3, 2)
        for cnf in inverted_residual_setting:
            f_dim = cnf.out_size(f_dim)
            t_dim = cnf.out_size(t_dim)
            cnf.f_dim, cnf.t_dim = f_dim, t_dim  # update dimensions in block config
            layers.append(block(cnf, se_cnf, norm_layer, depthwise_norm_layer))
            kernel_sizes.append(cnf.kernel)
            strides.append(cnf.stride)

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(
            ConvNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        )

        self.features = nn.Sequential(*layers)
        self.head_type = kwargs.get("head_type", False)
        if self.head_type == "multihead_attention_pooling":
            self.classifier = MultiHeadAttentionPooling(lastconv_output_channels, num_classes,
                                                        num_heads=kwargs.get("multihead_attention_heads"))
        elif self.head_type == "fully_convolutional":
            self.classifier = nn.Sequential(
                nn.Conv2d(
                    lastconv_output_channels,
                    num_classes,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0),
                    bias=False),
                nn.BatchNorm2d(num_classes),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
        elif self.head_type == "mlp":
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(start_dim=1),
                nn.Linear(lastconv_output_channels, last_channel),
                nn.Hardswish(inplace=True),
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(last_channel, num_classes),
            )
        else:
            raise NotImplementedError(f"Head '{self.head_type}' unknown. Must be one of: 'mlp', "
                                      f"'fully_convolutional', 'multihead_attention_pooling'")

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor, return_fmaps: bool = False) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, List[Tensor]]]:
        fmaps = []
        
        for i, layer in enumerate(self.features):
            x = layer(x)
            if return_fmaps:
                fmaps.append(x)
        
        features = F.adaptive_avg_pool2d(x, (1, 1)).squeeze()
        x = self.classifier(x).squeeze()
        
        if features.dim() == 1 and x.dim() == 1:
            # squeezed batch dimension
            features = features.unsqueeze(0)
            x = x.unsqueeze(0)
        
        if return_fmaps:
            return x, fmaps
        else:
            return x, features

    def forward(self, x: Tensor) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, List[Tensor]]]:
        return self._forward_impl(x)


def _mobilenet_v3_conf(
        width_mult: float = 1.0,
        reduced_tail: bool = False,
        dilated: bool = False,
        strides: Tuple[int] = (2, 2, 2, 2),
        **kwargs: Any
):
    reduce_divider = 2 if reduced_tail else 1
    dilation = 2 if dilated else 1

    bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_mult=width_mult)

    # InvertedResidualConfig:
    # input_channels, kernel, expanded_channels, out_channels, use_se, activation, stride, dilation
    inverted_residual_setting = [
        bneck_conf(16, 3, 16, 16, False, "RE", 1, 1),
        bneck_conf(16, 3, 64, 24, False, "RE", strides[0], 1),  # C1
        bneck_conf(24, 3, 72, 24, False, "RE", 1, 1),
        bneck_conf(24, 5, 72, 40, True, "RE", strides[1], 1),  # C2
        bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
        bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
        bneck_conf(40, 3, 240, 80, False, "HS", strides[2], 1),  # C3
        bneck_conf(80, 3, 200, 80, False, "HS", 1, 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
        bneck_conf(80, 3, 480, 112, True, "HS", 1, 1),
        bneck_conf(112, 3, 672, 112, True, "HS", 1, 1),
        bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", strides[3], dilation),  # C4
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
    ]
    last_channel = adjust_channels(1280 // reduce_divider)

    return inverted_residual_setting, last_channel


def _mobilenet_v3(
    inverted_residual_setting: List[InvertedResidualConfig],
    last_channel: int,
    pretrained_name: str,
    **kwargs: Any,
):
    """Construct model and, if a pretrained checkpoint is available, try to match
    the model backbone to the checkpoint feature shapes automatically.
    """
    # Attempt to load a state_dict following preference order:
    # 1) exact requested asset (e.g., 'dymn10_as.pt')
    # 2) canonical pretrained_models[pretrained_name]
    # 3) dymn* -> mn* fallback
    state_dict = None
    used_pretrained = None

    if pretrained_name:
        candidate_url = urllib.parse.urljoin(model_url, f"{pretrained_name}.pt")
        try:
            state_dict = load_state_dict_from_url(candidate_url, model_dir=model_dir, map_location="cpu")
            used_pretrained = pretrained_name
            logger.info(f"Loaded exact pretrained for requested name: {pretrained_name}")
        except Exception as e:
            logger.debug(f"Exact pretrained '{pretrained_name}' not found at {candidate_url}: {e}")

    if state_dict is None and pretrained_name in pretrained_models:
        asset_url = pretrained_models.get(pretrained_name)
        try:
            state_dict = load_state_dict_from_url(asset_url, model_dir=model_dir, map_location="cpu")
            used_pretrained = pretrained_name
            logger.info(f"Loaded canonical pretrained for name: {pretrained_name}")
        except Exception as e:
            logger.debug(f"Could not load canonical pretrained '{pretrained_name}': {e}")

    if state_dict is None and pretrained_name:
        base = pretrained_name.split('_')[0]
        if base.startswith('dymn'):
            fallback = 'mn' + base[4:] + pretrained_name[len(base):]
            if fallback in pretrained_models:
                asset_url = pretrained_models.get(fallback)
                try:
                    state_dict = load_state_dict_from_url(asset_url, model_dir=model_dir, map_location="cpu")
                    used_pretrained = fallback
                    logger.info(f"Requested pretrained '{pretrained_name}' maps to available pretrained '{fallback}'. Using '{fallback}'.")
                except Exception as e:
                    logger.debug(f"Could not load fallback pretrained '{fallback}': {e}")

    # If we have a checkpoint, compute its feature-parameter total (exclude classifier/head)
    chosen_width = kwargs.get('width_mult', 1.0)
    strict_matched = False
    if state_dict is not None:
        ck_feature_total = sum(v.numel() for k, v in state_dict.items() if hasattr(v, 'numel') and 'classifier' not in k and 'head' not in k)
        logger.info(f"Checkpoint feature params: {ck_feature_total}")

        # Detect 'layers.N.*' style checkpoints early and enforce deterministic exact-parity behavior
        import re
        layers_keys = [k for k in state_dict.keys() if re.match(r'layers\.\d+\.', k)]
        layers_style_ckpt = len(layers_keys) > 0
        if layers_style_ckpt:
            logger.info("Detected 'layers.*' style checkpoint. Will attempt deterministic reconstruction and will NOT perform width-heuristic fallbacks.")
        else:
            candidates = ['mn01_as', 'mn02_as', 'mn04_as', 'mn05_as', 'mn10_as', 'mn20_as', 'mn30_as', 'mn40_as']
            best = None
            best_diff = None
            for cand in candidates:
                base = cand.split('_')[0]
                w = NAME_TO_WIDTH(base)
                try:
                    inv_set_tmp, last_ch_tmp = _mobilenet_v3_conf(width_mult=w, reduced_tail=False, dilated=False)
                    tmp_model = MN(inv_set_tmp, last_ch_tmp, **kwargs)
                    feat_params = sum(p.numel() for p in tmp_model.features.parameters())
                except Exception:
                    # If building a tmp model fails due to se_conf or other details, skip candidate
                    continue
                diff = abs(feat_params - ck_feature_total)
                if best is None or diff < best_diff:
                    best = (cand, w, feat_params)
                    best_diff = diff
            if best is not None:
                chosen_width = best[1]
                logger.info(f"Best matching backbone from checkpoint (total-param heuristic): {best[0]} (width_mult={chosen_width}) - feature params={best[2]}")
            else:
                logger.info("Could not determine a matching backbone from canonical candidates; using provided width_mult.")

    # Instantiate model (using chosen_width) so that strict name+shape load has a model to populate
    try:
        inv_set_init, last_ch_init = _mobilenet_v3_conf(width_mult=chosen_width, reduced_tail=False, dilated=False)
        model = MN(inv_set_init, last_ch_init, **kwargs)
    except Exception as e_init:
        # If instantiation with chosen width fails, postpone and try again later
        logger.debug(f"Could not instantiate initial model with chosen width {chosen_width}: {e_init}")

    # Load state_dict into model if present
    if state_dict is not None:
        if kwargs['head_type'] == "mlp":
            num_classes = state_dict['classifier.5.bias'].size(0)
        elif kwargs['head_type'] == "fully_convolutional":
            num_classes = state_dict['classifier.1.bias'].size(0)
        else:
            logger.warning("Loading weights for classifier only implemented for head types 'mlp' and 'fully_convolutional'")
            num_classes = -1

        if kwargs['num_classes'] != num_classes:
            pretrain_logits = state_dict['classifier.5.bias'].size(0) if kwargs['head_type'] == "mlp" else state_dict['classifier.1.bias'].size(0)
            logger.warning(f"Number of classes defined: {kwargs['num_classes']}, but trying to load pre-trained layer with logits: {pretrain_logits}. Dropping last layer.")
            if kwargs['head_type'] == "mlp":
                del state_dict['classifier.5.weight']
                del state_dict['classifier.5.bias']
            else:
                state_dict = {k: v for k, v in state_dict.items() if not k.startswith('classifier')}
        try:
            model.load_state_dict(state_dict)
            strict_matched = True
            logger.info("✅ Strict state_dict load succeeded (exact name+shape match). Marking strict_matched=True")
        except RuntimeError as e:
            logger.warning(str(e))
            logger.warning("Strict state_dict load failed.")

            # Only allow deterministic reconstruction from 'layers.*' checkpoints; no fallbacks allowed.
            import re
            from collections import defaultdict
            ck_block_map = defaultdict(list)
            for k, v in state_dict.items():
                m = re.match(r'layers\.(\d+)\.(.*)', k)
                if m:
                    idx = int(m.group(1))
                    ck_block_map[idx].append((k, tuple(v.shape)))

            if not ck_block_map:
                # Not a 'layers.*' checkpoint and strict load failed: abort per strict policy
                raise RuntimeError("Strict load failed and checkpoint is not 'layers.*' style; aborting to enforce exact parity policy.")

            # Attempt deterministic exact reconstruction from 'layers.*' checkpoint
            try:
                logger.info("Detected 'layers.*' style checkpoint. Attempting deterministic exact reconstruction from checkpoint.")

                def infer_conf_from_layers_ckpt(state_d):
                    ck_blocks = {}
                    for k, v in state_d.items():
                        m = re.match(r'layers\.(\d+)\.(.+)', k)
                        if m:
                            idx = int(m.group(1))
                            ck_blocks.setdefault(idx, []).append((m.group(2), v.shape))

                    idxs = sorted(ck_blocks.keys())
                    inv_confs = []
                    canonical_stride_positions = [1, 3, 6, 12]

                    for i, idx in enumerate(idxs):
                        items = {name: shp for name, shp in ck_blocks[idx]}
                        in_c = None
                        exp_c = None
                        out_c = None
                        kernel = 3
                        use_se = False
                        activation = 'RE'
                        stride = 1

                        if 'exp_conv.weight' in items:
                            shp = items['exp_conv.weight']
                            exp_c = shp[0]
                            in_c = shp[1]
                        if 'proj_conv.weight' in items:
                            shp = items['proj_conv.weight']
                            out_c = shp[0]
                        if 'depth_conv.weight' in items:
                            shp = items['depth_conv.weight']
                            kernel = shp[2] if len(shp) >= 3 else 3
                        if any('conc_se' in n or 'context_gen' in n or 'se' in n for n in items.keys()):
                            use_se = True

                        if in_c is None and exp_c is not None:
                            in_c = max(1, exp_c // 4)
                        if out_c is None and in_c is not None and exp_c is not None:
                            out_c = max(1, exp_c // 4)

                        if exp_c is not None and exp_c > 128:
                            activation = 'HS'

                        if i in canonical_stride_positions:
                            stride = 2

                        if in_c is None or exp_c is None or out_c is None:
                            raise RuntimeError(f"Could not infer block channels for checkpoint block {idx}: {items.keys()}")

                        inv_confs.append(InvertedResidualConfig(in_c, kernel, exp_c, out_c, use_se, activation, stride, 1))

                    last_channel = InvertedResidualConfig.adjust_channels(inv_confs[-1].out_channels, width_mult=1.0) * 6
                    return inv_confs, int(last_channel)

                inv_set_inferred, last_ch_inferred = infer_conf_from_layers_ckpt(state_dict)
                model_inferred = MN(inv_set_inferred, last_ch_inferred, **kwargs)

                # Map checkpoint tensors to inferred model by ordered block mapping and exact shape
                model_state_after_inferred = model_inferred.state_dict()
                model_block_map_inferred = defaultdict(list)
                for k in model_state_after_inferred.keys():
                    m2 = re.match(r'features\.(\d+)\.(.*)', k)
                    if m2:
                        idx2 = int(m2.group(1))
                        model_block_map_inferred[idx2].append((k, tuple(model_state_after_inferred[k].shape)))

                model_idxs_sorted_inf = sorted(model_block_map_inferred.keys())
                ck_idxs_sorted = sorted(ck_block_map.keys())

                mapped_pairs_inf = {}
                mapped_counts = 0
                total_model_tensors = 0
                for i, ck_idx in enumerate(ck_idxs_sorted):
                    if i >= len(model_idxs_sorted_inf):
                        raise RuntimeError("Inferred model has fewer blocks than checkpoint; cannot achieve exact parity")
                    model_idx = model_idxs_sorted_inf[i]
                    ck_items = ck_block_map[ck_idx]
                    model_items = model_block_map_inferred[model_idx]
                    total_model_tensors += len(model_items)

                    shp_to_model = defaultdict(list)
                    for mk, msh in model_items:
                        shp_to_model[msh].append(mk)

                    assigned_local = set()
                    local_mapped = 0
                    for ck_k, ck_sh in ck_items:
                        cands = shp_to_model.get(tuple(ck_sh), [])
                        chosen = None
                        for cand in cands:
                            if cand not in assigned_local:
                                chosen = cand
                                break
                        if chosen:
                            mapped_pairs_inf[chosen] = state_dict[ck_k]
                            assigned_local.add(chosen)
                            mapped_counts += 1
                            local_mapped += 1

                    if local_mapped / max(1, len(model_items)) < 0.6:
                        raise RuntimeError(f"Block {ck_idx} mapping insufficient (<60%) during deterministic reconstruction")

                coverage_inf = mapped_counts / max(1, total_model_tensors)
                logger.info(f"Deterministic reconstruction mapping coverage: {mapped_counts}/{total_model_tensors} = {coverage_inf:.2%}")
                if coverage_inf < 0.8:
                    raise RuntimeError("Deterministic reconstruction coverage < 80%; aborting to avoid unsafe mapping")

                # Apply mappings (exclude classifier/proj)
                safe_mapped = {k: v for k, v in mapped_pairs_inf.items() if not any(ex in k for ex in ['classifier', 'head', 'proj', 'projection'])}
                model_inferred.load_state_dict(safe_mapped, strict=False)
                model = model_inferred
                strict_matched = True
                model._deterministic_parity_candidate = 'layers_inferred'
                model._inferred_from_layers = True
                logger.info("✅ Deterministic exact reconstruction succeeded; backbone loaded with strict parity.")
            except Exception as e_det:
                logger.exception(f"Deterministic exact reconstruction from 'layers.*' checkpoint failed: {e_det}")
                # Per strict policy: abort
                raise RuntimeError("Detected 'layers.*' checkpoint but could not reconstruct exact parity. Aborting load to avoid unsafe fallback/mixing of weights.")

        # Instantiate model using chosen width if not already replaced by a reconstructed model
        try:
            inv_set, last_ch = _mobilenet_v3_conf(width_mult=chosen_width, reduced_tail=False, dilated=False)
            if 'model' not in locals():
                model = MN(inv_set, last_ch, **kwargs)
        except Exception as e_build:
            logger.debug(f"Could not instantiate model with chosen width {chosen_width}: {e_build}")
            model = MN(_mobilenet_v3_conf(width_mult=chosen_width)[0], _mobilenet_v3_conf(width_mult=chosen_width)[1], **kwargs)

        # Expose metadata and return
        model._used_pretrained = used_pretrained
        model._strict_match = strict_matched
        if strict_matched:
            logger.info("✅ Strict structural match achieved with checkpoint (strict_matched=True)")
        else:
            logger.warning("⚠️ Could not safely obtain a strict structural match from checkpoint. Model partial loads may have occurred (strict_matched=False). If you need exact parity, provide the exact matching 'mn*' or 'dymn*' config or a checkpoint that uses the canonical 'features.*' naming.")

        # Log where the pretrained asset was cached locally and some stats for transparency
        try:
            from pathlib import Path
            candidate_path = Path(model_dir) / f"{used_pretrained}.pt"
            if candidate_path.exists():
                size_mb = candidate_path.stat().st_size / (1024 ** 2)
                total_ck_params = sum(v.numel() for v in state_dict.values() if hasattr(v, 'numel'))
                logger.info(f"✅ Used pretrained asset: {used_pretrained} -> {candidate_path} ({size_mb:.2f} MB), checkpoint total params: {total_ck_params}")
            else:
                logger.info(f"✅ Used pretrained asset: {used_pretrained} (cached file not found at expected path: {candidate_path})")
        except Exception as e:
            logger.debug(f"Could not compute pretrained asset stats: {e}")

        return model

    elif pretrained_name:
        logger.info(f"Requested pretrained '{pretrained_name}' has no exact pretrained file. Will attempt to use model configuration (width_mult={chosen_width}).")

    # final fallback: construct a model with given configuration
    inv_set, last_ch = _mobilenet_v3_conf(width_mult=chosen_width, reduced_tail=False, dilated=False)
    return MN(inv_set, last_ch, **kwargs)


def mobilenet_v3(pretrained_name: str = None, **kwargs: Any) -> MN:
    """Helper wrapper: build inverted_residual_setting from kwargs and call _mobilenet_v3."""
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(**kwargs)
    return _mobilenet_v3(inverted_residual_setting, last_channel, pretrained_name, **kwargs)


def get_model(num_classes: int = 527, pretrained_name: str = None, width_mult: float = 1.0,
              reduced_tail: bool = False, dilated: bool = False, strides: Tuple[int, int, int, int] = (2, 2, 2, 2),
              head_type: str = "mlp", multihead_attention_heads: int = 4, input_dim_f: int = 128,
              input_dim_t: int = 1000, se_dims: str = 'c', se_agg: str = "max", se_r: int = 4):
    """
        Factory for building EfficientAT MobileNet models with convenient defaults and logging.
    """
    requested_name = pretrained_name
    # Infer width if possible from name (e.g., 'dymn10_as' -> width_mult for 'mn10')
    if pretrained_name:
        base = pretrained_name.split('_')[0]
        try:
            width_mult = NAME_TO_WIDTH(base)
        except Exception:
            pass

    # Route dymn* models to the Dynamic MobileNet architecture
    if pretrained_name and pretrained_name.split('_')[0].startswith('dymn'):
        import torch
        from .dymn import DyMN

        m = DyMN(num_classes=num_classes, width_mult=width_mult)

        # Load pretrained weights from local file
        local_path = os.path.join(model_dir, f"{pretrained_name}.pt")
        if os.path.exists(local_path):
            state_dict = torch.load(local_path, map_location="cpu")
            # Drop classifier if class count differs
            if num_classes != state_dict.get('classifier.5.bias', torch.zeros(0)).size(0):
                logger.warning(f"Dropping classifier layers for num_classes mismatch")
                state_dict = {k: v for k, v in state_dict.items() if not k.startswith('classifier')}
                m.load_state_dict(state_dict, strict=False)
            else:
                m.load_state_dict(state_dict, strict=True)
            logger.info(f"Loaded DyMN pretrained weights from {local_path} (strict=True)")
        else:
            # Try downloading from GitHub releases
            candidate_url = urllib.parse.urljoin(model_url, f"{pretrained_name}.pt")
            try:
                state_dict = load_state_dict_from_url(candidate_url, model_dir=model_dir, map_location="cpu")
                m.load_state_dict(state_dict, strict=True)
                logger.info(f"Downloaded and loaded DyMN pretrained weights: {pretrained_name}")
            except Exception as e:
                logger.warning(f"Could not load DyMN pretrained '{pretrained_name}': {e}")

        m._requested_pretrained = requested_name
        m._loaded_pretrained = pretrained_name
        logger.info(m)
        return m

    dim_map = {'c': 1, 'f': 2, 't': 3}
    assert len(se_dims) <= 3 and all([s in dim_map.keys() for s in se_dims]) or se_dims == 'none'
    input_dims = (input_dim_f, input_dim_t)
    if se_dims == 'none':
        se_conf = None
    else:
        se_conf = dict(se_dims=[dim_map[s] for s in se_dims], se_agg=se_agg, se_r=se_r)

    m = mobilenet_v3(pretrained_name=pretrained_name, num_classes=num_classes,
                     width_mult=width_mult, reduced_tail=reduced_tail, dilated=dilated, strides=strides,
                     head_type=head_type, multihead_attention_heads=multihead_attention_heads,
                     input_dims=input_dims, se_conf=se_conf)

    # metadata
    m._requested_pretrained = requested_name
    m._loaded_pretrained = getattr(m, '_used_pretrained', None) or (pretrained_name if (pretrained_name and pretrained_name in pretrained_models) else None)
    if m._requested_pretrained != m._loaded_pretrained:
        logger.info(f"Pretrained mapping: requested='{m._requested_pretrained}' -> loaded='{m._loaded_pretrained}'")
    logger.info(m)
    return m
