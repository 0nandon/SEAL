import os
import torch
import tqdm
from loss.loss_function import TotalLoss
from data_utils.eventsam_loader import RGBEData
from models.build_mix_rgbe_encoder import _build_mix_rgbe_encoder_b


def run():
    print('=====Training script for event_vit_encoder')
    # Build dataloaders
    RGBEDataset = RGBEData(
        root='dataset/metadata',
        data='train.txt',
        img_width=512
    )
    RGBELoader = torch.utils.data.DataLoader(
        dataset=RGBEDataset, 
        batch_size=24, 
        shuffle=True
    )

    # Create the teacher and student encoders
    RGBE_Encoder = _build_mix_rgbe_encoder_b(
        checkpoint_path="cache/sam_vit_b_01ec64.pth"
    )

    # MultiGPU Train
    RGBE_Encoder = torch.nn.DataParallel(RGBE_Encoder).cuda()

    for name, param in RGBE_Encoder.named_parameters():
        param.requires_grad = False

    train_block_list = ["evimg_encoder.patch_embed"] + ["evimg_encoder.blocks." + str(i) + ".mlp" for i in [2, 5, 8, 11]]
    for name, param in RGBE_Encoder.named_parameters():
        for block_name in train_block_list:
            if block_name in name:
                param.requires_grad = True

    train_para_name_list = []
    for name, param in RGBE_Encoder.named_parameters():
        if param.requires_grad:
            train_para_name_list.append(name)

    print("=====Train Params:",train_para_name_list)
    # # optimizer
    optimizer = torch.optim.Adam(params=[{'params': [p for name, p in RGBE_Encoder.named_parameters() if name in train_para_name_list]}], lr=2e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)
    # Loss functions
    loss_fun = TotalLoss(block_feature_indexes=[2, 5, 8, 11], block_loss_weight=[0.1, 0.4, 0.7, 1.0])

    iteration = 0
    size = 0
    RGBE_Encoder.train()
    for epoch in range(5):
        loss_sum = 0.0
        source_loss_sum_list = [0.0, 0.0, 0.0, 0.0,0.0]
        loss_sum_list = [0.0, 0.0, 0.0, 0.0,0.0]
        print('=====epoch ' + str(epoch))
        for images, evimgs in RGBELoader:
            images = images.cuda()
            evimgs = evimgs.cuda()
            optimizer.zero_grad()
            
            if images.shape[0] % 2 == 1:
                images = images[:-1, :, :, :]
                evimgs = evimgs[:-1, :, :, :]
                
            image_embeddings_dict, evimg_embeddings_dict, token_weights_dict = RGBE_Encoder(images, evimgs)

            loss,source_loss_list,loss_list = loss_fun(
                image_embeddings_dict, 
                evimg_embeddings_dict, 
                token_weights_dict,
                images,
                evimgs
            )

            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            source_loss_sum_list = [x + y.item() for x, y in zip(source_loss_sum_list, source_loss_list)]
            loss_sum_list = [x + y.item() for x, y in zip(loss_sum_list, loss_list)]

            iteration += 1
            size += 1

            if iteration % 10 == 0:
                print('iteration:', iteration, 'loss:', loss_sum / 10, "source_loss_list:",[x/10 for x in source_loss_sum_list], 'loss_list:', [x / 10 for x in loss_sum_list])
                loss_sum = 0.0
                source_loss_sum_list = [0.0, 0.0, 0.0, 0.0, 0.0]
                loss_sum_list = [0.0, 0.0, 0.0, 0.0, 0.0]
        
        torch.save(RGBE_Encoder.state_dict(), 'eventsam_checkpoints/rgbe_encoder_0%d_iter.pth' % (epoch))

        if epoch % 3 == 0:
            scheduler.step()
        print('iteration:', iteration)
        print('loss:', loss_sum/size)
        size = 0


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    run()

