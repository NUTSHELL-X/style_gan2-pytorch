import configargparse

def config_parser():
    parser=configargparse.ArgumentParser()
    parser.add_argument('--config',is_config_file=True,help='config file path',default='config.txt')
    parser.add_argument('--device',type=str,default='cuda:0')
    parser.add_argument('--model_path',type=str) # path to save weights(.pt file)
    parser.add_argument('--generated_image_folder',type=str) # path to save generated images(.jpg file) 
    parser.add_argument('--dataset_path',type=str) # path containing dataset
    parser.add_argument('--dataset_type',type=str,default='image_folder') # path containing dataset
    parser.add_argument('--epoch',type=int)
    parser.add_argument('--dtype',type=str) # data type used for training(fp16,fp32,mixed)
    parser.add_argument('--start_res',nargs='+',type=int) # 
    parser.add_argument('--upscale_times',type=int)
    parser.add_argument('--start_c',type=int) # channels for input constant(torch.ones)
    parser.add_argument('--w_c',type=int) # channels for style code
    parser.add_argument('--batch_size',type=int)
    parser.add_argument('--lr',type=float) # learning rate
    parser.add_argument('--lr_decay',type=bool) # learning rate decay or not
    parser.add_argument('--milestones',type=int,nargs='+',action='append') # milestones for learning rate decay
    parser.add_argument('--gamma',type=float) # learning rate decay ratio

    return parser