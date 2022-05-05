import os
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path',type=str,help="data path")
parser.add_argument('--max_epoch',type=int,default=400,help="data path")
args = parser.parse_args()

def read_num(path):
    with open(os.path.join(path,'data_size.txt')) as f:
        message = f.readline()
        user_num, item_num = message.strip().split()
        user_num, item_num = int(user_num), int(item_num)
        print(user_num, item_num)
    return user_num, item_num

def generate_dict(path):
    user_interaction = {}
    with open(os.path.join(path,'train.txt')) as f:
        data = f.readlines()
        for row in data:
            user, item = row.strip().split()
            user, item = int(user), int(item)
            
            if user not in user_interaction:
                user_interaction[user]=[item]
            elif item not in user_interaction[user]:
                user_interaction[user].append(item)
    return user_interaction

def sample():
    user_num, item_num = read_num(args.path)
    user_interaction = generate_dict(args.path)

    for i in range(args.max_epoch):
        print('Round {} Start!'.format(i))
        with open(os.path.join(args.path, 'sample_file','sample_'+str(i)+'.txt'),'w') as f1:
            with open(os.path.join(args.path,'train.txt')) as f:
                data = f.readlines()
                for row in data:
                    user, item = row.strip().split()
                    user, item = int(user), int(item)
                        
                    while True:
                        rand_item = random.randint(0,item_num-1)
                        if rand_item not in user_interaction[user]:
                            break
                    f1.write(str(user)+' '+str(item)+' '+str(rand_item)+'\n')

if __name__=='__main__':
    sample()

# python sample.py --path=/data3/jinbowen/multi_behavior/data/Tmall_test --max_epoch=20
