import configargparse
import torch

def config_parser():
    parser=configargparse.ArgumentParser()
    parser.add_argument('--config',is_config_file=True,help='config file path',default='config.txt')
    parser.add_argument('--device',type=str,default='cuda')
    parser.add_argument('--model_path',type=str) # path to save weights(.pt file)
    parser.add_argument('--gen_weights_path',type=str) # path to save generator
    parser.add_argument('--disc_weights_path',type=str) # path to save discriminator
    parser.add_argument('--training_params_path',type=str) # path to save training parameters(optimizor,total training epochs)
    parser.add_argument('--save_dir',type=str,default='./exp') # 
    parser.add_argument('--save_images',type=bool,default=True) # whether save images while training or not
    parser.add_argument('--generated_image_folder',type=str) # path to save generated images(.jpg file) 
    parser.add_argument('--save_freq_inside_epoch',type=int) # frequency to save image inside a epoch
    parser.add_argument('--dataset_path',type=str) # path containing dataset
    parser.add_argument('--dataset_type',type=str,default='image_folder') # dataset type (see dataset.py)
    parser.add_argument('--loss_fn_gen',type=str,default='BCE')
    parser.add_argument('--loss_fn_disc',type=str,default='BCE')
    parser.add_argument('--epochs',type=int) # training epochs
    parser.add_argument('--dtype',type=str) # data type used for training(float16,float32)
    parser.add_argument('--start_res',nargs='+',type=int) # start resolution 
    parser.add_argument('--upscale_times',type=int) # times of 2x upscaling
    parser.add_argument('--start_channels',type=int) # channels for input constant(torch.ones)
    parser.add_argument('--w_channels',type=int) # channels for style code
    parser.add_argument('--batch_size',type=int) # batch size
    parser.add_argument('--lr',type=float) # learning rate
    parser.add_argument('--lr_decay',type=bool) # learning rate decay or not
    parser.add_argument('--lr_step_size',type=int) # lr decay frequency
    parser.add_argument('--milestones',type=int,nargs='+',action='append') # milestones for learning rate decay
    parser.add_argument('--gamma',type=float) # learning rate decay ratio
    parser.add_argument('--continues',type=bool,default=True) # continues training or not
    parser.add_argument('--gpus',nargs='+',type=int) # used gpus e.g [0,1]

    return parser

if __name__=='__main__': # show all args
    parser=config_parser()
    args=parser.parse_args()
    print(args)
    for arg in vars(args):
        value = getattr(args,arg)
        print('arg: ',arg,'value: ',value,'type: ',type(value))