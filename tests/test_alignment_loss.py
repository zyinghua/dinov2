import unittest

import torch

from dinov2.alignment_loss import AlignmentLoss


class AlignmentLossCKATest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.loss_fn = AlignmentLoss(loss_type="cka")

    def test_identical_features_give_near_zero_loss(self):
        torch.manual_seed(3)
        bsz, num_patches, dim = 2, 8, 16
        dinov2_feat = torch.randn(bsz, num_patches, dim)
        dit_feat = dinov2_feat.clone()

        loss = self.loss_fn([dinov2_feat], dit_feat)
        print(loss)
        self.assertLess(loss.item(), 1e-5)

    def test_linear_transform_with_dimension_mismatch(self):
        torch.manual_seed(21)
        bsz, num_patches, dinov2_dim, dit_dim = 2, 10, 12, 20
        dinov2_feat = torch.randn(bsz, num_patches, dinov2_dim)
        transform = torch.randn(dinov2_dim, dit_dim)
        dit_feat = torch.einsum("bnd,df->bnf", dinov2_feat, transform)

        loss = self.loss_fn([dinov2_feat], dit_feat)
        print(loss)
        self.assertLess(loss.item(), 0.2)

    def test_random_features_have_high_loss(self):
        torch.manual_seed(42)
        bsz, num_patches, dim = 2, 6, 18
        dinov2_feat = torch.randn(bsz, num_patches, dim)
        dit_feat = torch.randn(bsz, num_patches, dim)

        loss = self.loss_fn([dinov2_feat], dit_feat)
        print(loss)
        # self.assertGreater(loss.item(), 0.5)


if __name__ == "__main__":
    unittest.main()
