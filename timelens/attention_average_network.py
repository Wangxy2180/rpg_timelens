import torch as th
import torch.nn.functional as F
from timelens import refine_warp_network, warp_network
from timelens.superslomo import unet


def _pack_input_for_attention_computation(example):
    fusion = example["middle"]["fusion"]
    number_of_examples, _, height, width = fusion.size()
    return th.cat(
        [
            example["after"]["flow"],
            example["middle"]["after_refined_warped"],
            example["before"]["flow"],
            example["middle"]["before_refined_warped"],
            example["middle"]["fusion"],
            th.Tensor(example["middle"]["weight"])
                .view(-1, 1, 1, 1)
                .expand(number_of_examples, 1, height, width)
                .type(fusion.type()),
        ],
        dim=1,
    )


def _compute_weighted_average(attention, before_refined, after_refined, fusion):
    return (
            attention[:, 0, ...].unsqueeze(1) * before_refined
            + attention[:, 1, ...].unsqueeze(1) * after_refined
            + attention[:, 2, ...].unsqueeze(1) * fusion
    )


class AttentionAverage(refine_warp_network.RefineWarp):
    def __init__(self):
        warp_network.Warp.__init__(self)
        # 这里前两个网络也没用上啊，还是直接调用了父类
        # 输出channel和输出channel
        self.fusion_network = unet.UNet(2 * 3 + 2 * 5, 3, False)
        self.flow_refinement_network = unet.UNet(9, 4, False)
        self.attention_network = unet.UNet(14, 3, False)

    def run_fast(self, example):
        # 这个意思！！莫非是先用warp，然后再用synthesis？
        example['middle']['before_refined_warped'], \
        example['middle']['after_refined_warped'] = refine_warp_network.RefineWarp.run_fast(self, example)
        # example真是大扩充啊
        # {mid:1,before:6,after:4}->{{mid:8,before:7,after:5}}
        attention_scores = self.attention_network(
            _pack_input_for_attention_computation(example)
        )
        attention = F.softmax(attention_scores, dim=1)
        average = _compute_weighted_average(
            attention,
            example['middle']['before_refined_warped'],
            example['middle']['after_refined_warped'],
            example['middle']['fusion']
        )
        # 这个average就是生成的图片了
        return average, attention

    def run_attention_averaging(self, example):
        refine_warp_network.RefineWarp.run_and_pack_to_example(self, example)
        attention_scores = self.attention_network(
            _pack_input_for_attention_computation(example)
        )
        attention = F.softmax(attention_scores, dim=1)
        average = _compute_weighted_average(
            attention,
            example["middle"]["before_refined_warped"],
            example["middle"]["after_refined_warped"],
            example["middle"]["fusion"],
        )
        return average, attention

    def run_and_pack_to_example(self, example):
        (
            example["middle"]["attention_average"],
            example["middle"]["attention"],
        ) = self.run_attention_averaging(example)

    def forward(self, example):
        return self.run_attention_averaging(example)
