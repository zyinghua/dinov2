import torch
import torch.nn as nn


################### Classifier ############################# 


def create_linear_input(x_tokens_list, use_n_blocks, use_avgpool):
    if torch.is_tensor(x_tokens_list):
        return x_tokens_list.float()
    intermediate_output = x_tokens_list[-use_n_blocks:]
    output = torch.cat([class_token for _, class_token in intermediate_output], dim=-1)
    if use_avgpool:
        output = torch.cat(
            (
                output,
                torch.mean(intermediate_output[-1][0], dim=1),  # patch tokens
            ),
            dim=-1,
        )
        output = output.reshape(output.shape[0], -1)
    return output.float()


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""

    def __init__(self, out_dim, use_n_blocks, use_avgpool, num_classes=1000):
        super().__init__()
        self.out_dim = out_dim
        self.use_n_blocks = use_n_blocks
        self.use_avgpool = use_avgpool
        self.num_classes = num_classes
        self.linear = nn.Linear(out_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x_tokens_list):
        output = create_linear_input(x_tokens_list, self.use_n_blocks, self.use_avgpool)
        return self.linear(output)


class AllClassifiers(nn.Module):
    def __init__(self, classifiers_dict):
        super().__init__()
        self.classifiers_dict = nn.ModuleDict()
        self.classifiers_dict.update(classifiers_dict)

    def forward(self, inputs):
        return {k: v.forward(inputs) for k, v in self.classifiers_dict.items()}

    def __len__(self):
        return len(self.classifiers_dict)


class LinearPostprocessor(nn.Module):
    def __init__(self, linear_classifier, class_mapping=None):
        super().__init__()
        self.linear_classifier = linear_classifier
        self.register_buffer("class_mapping", None if class_mapping is None else torch.LongTensor(class_mapping))

    def forward(self, samples, targets):
        preds = self.linear_classifier(samples)
        return {
            "preds": preds[:, self.class_mapping] if self.class_mapping is not None else preds,
            "target": targets,
        }


def scale_lr(learning_rates, batch_size):
    return learning_rates * (batch_size) / 256.0


def setup_linear_classifiers(sample_output, n_last_blocks_list, learning_rates, batch_size, num_classes=1000):
    linear_classifiers_dict = nn.ModuleDict()
    optim_param_groups = []
    for n in n_last_blocks_list:
        for avgpool in [False, True]:
            for _lr in learning_rates:
                lr = scale_lr(_lr, batch_size)
                out_dim = create_linear_input(sample_output, use_n_blocks=n, use_avgpool=avgpool).shape[1]
                linear_classifier = LinearClassifier(
                    out_dim, use_n_blocks=n, use_avgpool=avgpool, num_classes=num_classes
                )
                linear_classifier = linear_classifier.cuda()
                linear_classifiers_dict[
                    f"classifier_{n}_blocks_avgpool_{avgpool}_lr_{lr:.5f}".replace(".", "_")
                ] = linear_classifier
                optim_param_groups.append({"params": linear_classifier.parameters(), "lr": lr})

    linear_classifiers = AllClassifiers(linear_classifiers_dict)


    return linear_classifiers, optim_param_groups