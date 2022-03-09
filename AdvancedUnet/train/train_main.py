from util.train_utils import *
import wandb
import logging
from loss import mul_bce_loss_fusion

# paths
root_path = '..'
train_tag = 'demo_emojis'

# datasets paths
# cache_root = ['data/train/demo_text_vm_ds', 'data folder b', '...']
cache_root = ['../data/train/demo_text_vm_ds']

# dataset configurations
patch_size = None
image_size = 512

# network
nets_path = '%s/checkpoints/%s' % (root_path, train_tag)
images_path = '%s/images' % nets_path

num_blocks = (3, 3, 3, 3, 3)
shared_depth = 2
use_vm_decoder = True

# train configurations
gamma1 = 2  # L1 image
gamma2 = 1  # L1 visual motif
epochs = 20
batch_size = 1
print_frequency = 100
save_frequency = 10
device = torch.device('cuda:0')


def l1_relative(reconstructed, real, batch, area):
    loss_l1 = torch.abs(reconstructed - real).view(batch, -1)
    loss_l1 = torch.sum(loss_l1, dim=1) / area
    loss_l1 = torch.sum(loss_l1) / batch
    return loss_l1


def wanInit():
    experiment = wandb.init(project="demo", resume=True, entity="breezewrf")
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size,
                                  save_checkpoint=True))
    logging.info(f'''Starting training:
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Checkpoints:     {True}
            Device:          {device.type}
        ''')
    return experiment


def train(net, train_loader, test_loader, log=False):
    if log:
        experiment = wanInit()
    net.set_optimizers()
    losses = []
    print('Training Begins')
    for epoch in range(epochs):
        real_epoch = epoch + 1
        for i, data in enumerate(train_loader):
            """
            synthesized: 合成水印图像
            images: 原图
            vm_mask: 水印掩膜
            motifs: 水印彩色
            vm_area: 水印面积（这里没用到）
            """
            synthesized, images, vm_mask, motifs, vm_area = data
            synthesized, images, = synthesized.to(device), images.to(device)
            vm_mask, vm_area = vm_mask.to(device), vm_area.to(device)

            results = net(synthesized)
            """guess_images, guess_masks各有6张预测图，最后1张是前5张融合得到"""
            guess_images, guess_masks = results[0], results[1]
            expanded_vm_mask = vm_mask.repeat(1, 3, 1, 1)  # 通道扩张后才能*
            batch_cur_size = vm_mask.shape[0]
            loss_l1_images = []
            net.zero_grad_all()
            for g_image in guess_images:
                recon_pixels = g_image * expanded_vm_mask
                real_pixels = images * expanded_vm_mask
                loss_l1_images.append(l1_relative(recon_pixels, real_pixels, batch_cur_size, vm_area))
            loss_l1_image = sum(loss_l1_images) / len(loss_l1_images)

            loss_0, loss_mask = mul_bce_loss_fusion(guess_masks[0], guess_masks[1],
                                                    guess_masks[2], guess_masks[3],
                                                    guess_masks[4], guess_masks[5], vm_mask)
            loss_l1_vm = 0
            # 不采用loss_vm
            # if len(results) == 3:
            #     guess_vm = results[2]
            #     reconstructed_motifs = guess_vm * expanded_vm_mask
            #     real_vm = motifs.to(device) * expanded_vm_mask
            #     loss_l1_vm = l1_relative(reconstructed_motifs, real_vm, batch_cur_size, vm_area)
            loss = loss_l1_image + loss_mask
            loss.backward()
            net.step_all()
            losses.append(loss.item())
            # del temporary outputs
            del guess_images, guess_masks
            if log:
                experiment.log({"train_loss": sum(losses) / len(losses),
                                "step": i,
                                "epoch": epoch})
            # print
            if (i + 1) % print_frequency == 0:
                print('%s [%d, %3d] , baseline loss: %.2f' % (
                    train_tag, real_epoch, batch_size * (i + 1), sum(losses) / len(losses)))
                losses = []

        # savings
        if real_epoch % save_frequency == 0:
            print("checkpointing...")
            image_name = '%s/%s_%d.png' % (images_path, train_tag, real_epoch)
            _ = save_test_images(net, test_loader, image_name, device)
            torch.save(net.state_dict(), '%s/net_baseline.pth' % nets_path)
            torch.save(net.state_dict(), '%s/net_baseline_%d.pth' % (nets_path, real_epoch))

    print('Training Done:)')


def run():
    init_folders(nets_path, images_path)
    opt = load_globals(nets_path, globals(), override=True)
    train_loader, test_loader = init_loaders(opt, cache_root=cache_root)
    base_net = init_nets(opt, nets_path, device)
    train(base_net, train_loader, test_loader)


if __name__ == '__main__':
    run()
