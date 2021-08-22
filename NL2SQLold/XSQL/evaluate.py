import torch
import args
import numpy as np

def evaluate_(model, dev_data):
    total, losses = 0.0, []
    device = args.device

    with torch.no_grad():
        model.eval()
        for batch in dev_data:

            input_ids, input_mask,segment_ids, start_positions, end_positions = batch.input_ids, batch.input_mask, batch.segment_ids, batch.start_position, batch.end_position
            loss, _, _ = model(input_ids.to(device),segment_ids.to(device), input_mask.to(device), start_positions.to(device), end_positions.to(device))
            loss = loss / args.gradient_accumulation_steps
            losses.append(loss.item())

        for i in losses:
            total += i
        with open("./log", 'a') as f:
            f.write("eval_loss: " + str(total / len(losses)) + "\n")


# def find_best_answer(batch)



def evaluate(model,dev_data):
    device = args.device
    with torch.no_grad():
        model.eval()
        pred_answers,ref_answers = [],[]
        acc = 0
        whole = 0
        for batch in dev_data:
            input_ids, input_mask,segment_ids, start_positions, end_positions = batch.input_ids, batch.input_mask, batch.segment_ids, batch.start_position, batch.end_position
            # start_positions_prob,end_positions_prob = model(input_ids.to(device),segment_ids.to(device), input_mask.to(device), start_positions.to(device), end_positions.to(device))
            start_positions_prob, end_positions_prob = model(input_ids.to(device),segment_ids.to(device), input_mask.to(device))
            pred_start,index_start = torch.topk(start_positions_prob,k = 2,dim = -1, largest = True, sorted = True)
            end_positions_prob, end_positions_prob = model(input_ids.to(device),segment_ids.to(device), input_mask.to(device))
            pred_end,index_end = torch.topk(end_positions_prob,k = 2,dim = -1, largest = True, sorted = True)
            # print(start_positions)                            #tensor([1, 1, 1, 1])
            # print(end_positions)                              #tensor([2, 2, 2, 2])
            # print(index_start)                                #[[21, 43],[ 6, 52],[ 1,  2],[ 4,  2]]  
            # print(index_end)                                  #tensor([[19, 20],[11,  2],[ 8, 19],[ 7, 21]], device='cuda:0')
            start_positions_np = start_positions.numpy()
            end_positions_np = end_positions.numpy()
            index_start_np = index_start.cpu().numpy()
            index_end_np = index_end.cpu().numpy()

            # print(input_ids)
            # print(segment_ids)
            # print(input_mask)
            # exit()

            start_acc = [1 if id in list(index_start_np[i]) else 0 for i, id in enumerate(list(start_positions_np))]
            end_acc = [1 if id in list(index_end_np[i]) else 0 for i, id in enumerate(list(end_positions_np))]
            acc_list = [1 if start and end else 0 for start, end in zip(start_acc, end_acc)]
            
            # print()
            acc += sum(acc_list)
            whole += len(acc_list)
            # print(acc_list)
            # acc += acc_list
    return 'acc',float(acc/whole)
            # print(acc_list,start_acc,end_acc)
            # exit()
        # return total / len(losses)
            # [batch_size,max_length]

            # for index,eachquery in enumerate(batch):
            #     start_positions_prob = start_positions_prob[index].data.cpu().numpy()
            #     start_prob_topN =  np.argsort(-start_positions_prob)[:3]
            #     # end_positions_prob = end_positions_prob.data[index].cpu().numpy()
            #     # end_prob_topN =  np.argsort(-end_positions_prob[index])[:3]

            #     # print(start_prob_topN)
            #     # print(end_prob_topN)

            #     print(start_positions_prob)
            #     print(start_prob_topN)
            #     print(start_positions[index])
            #     exit()


