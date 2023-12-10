import configargparse

def config_parser():
    parser=configargparse.ArgumentParser()
    parser.add_argument('--config',is_config_file=True,help='config file path',default='config.txt')
    parser.add_argument('--device',type=str,default='cuda')
    parser.add_argument('--model_path',type=str) # path to save weights(.pt file)
    parser.add_argument('--gen_weights_path',type=str)
    parser.add_argument('--disc_weights_path',type=str)
    parser.add_argument('--training_params_path',type=str)
    parser.add_argument('--save_dir',type=str,default='./exp')
    parser.add_argument('--generated_image_folder',type=str) # path to save generated images(.jpg file) 
    parser.add_argument('--dataset_path',type=str) # path containing dataset
    parser.add_argument('--dataset_type',type=str,default='image_folder') # path containing dataset
    parser.add_argument('--epochs',type=int)
    parser.add_argument('--dtype',type=str) # data type used for training(fp16,fp32,mixed)
    parser.add_argument('--start_res',nargs='+',type=int) # 
    parser.add_argument('--upscale_times',type=int)
    parser.add_argument('--start_channels',type=int) # channels for input constant(torch.ones)
    parser.add_argument('--w_channels',type=int) # channels for style code
    parser.add_argument('--batch_size',type=int)
    parser.add_argument('--lr',type=float) # learning rate
    parser.add_argument('--lr_decay',type=bool) # learning rate decay or not
    parser.add_argument('--lr_step_size',type=int)
    parser.add_argument('--milestones',type=int,nargs='+',action='append') # milestones for learning rate decay
    parser.add_argument('--gamma',type=float) # learning rate decay ratio
    parser.add_argument('--continues',type=bool,default=False)
    parser.add_argument('--gpus',nargs='+',type=int)

    return parser

if __name__=='__main__':
    parser=config_parser()
    args=parser.parse_args()
    print(args)
    for arg in vars(args):
        value = getattr(args,arg)
        print('arg: ',arg,'value: ',value,'type: ',type(value))