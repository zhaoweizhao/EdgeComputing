def eval_inference(model, dataloader):
        model.eval()
        n_stage = len(model.module.blocks)
        exp = torch.zeros(n_stage)
        
        logits = [[] for _ in range(n_stage)]
        targets = []
        acc = 0
        n_sample = 0
        for i, (input, target) in enumerate(dataloader):
            n_sample += len(target)
            print(f"{i}/{len(dataloader)}")
            input = input.cuda()
            target = target.cuda()
            target = target.cpu()
            targets.append(target)
            with torch.no_grad():
                input_var = torch.autograd.Variable(input)
               
                final_logits, intermediate_logits, Gates = model.module.forward_for_inference(input_var)
                Gates = torch.cat(Gates, dim=1)
                # print(Gates.shape)
                # print(Gates)
                # print(final_logits.shape)
                actual_exits_binary = torch.nn.functional.sigmoid(Gates) 
                # print(actual_exits_binary)
                intermediate_logits.append(final_logits)
                output = intermediate_logits
                for i in range(final_logits.size(0)):
                    for j in range(n_stage):
                        if j < n_stage - 1:
                            if actual_exits_binary[i][j] >= 0.7: #确定退出
                                pred = output[j]
                                
                                pred = pred[i, :]
                                # print(pred.shape)
                                # _top_max_k_vals, top_max_k_inds = torch.topk(pred, 1, largest=True, sorted=True)
                                max_preds, argmax_preds = pred.max(dim=0, keepdim=False)
                                # print(argmax_preds)
                                # print(top_max_k_inds)
                                if argmax_preds == target[i]:
                                    acc += 1
                                exp[j] += 1
                                break
                        else:
                            pred = output[j]
                            pred = pred[i, :]
                            max_preds, argmax_preds = pred.max(dim=0, keepdim=False)
                    
                            if argmax_preds == target[i]:
                                acc += 1
                            exp[j] += 1

        exit_rate = [0] * n_stage
        print(n_sample)
        print(acc)
        for k in range(n_stage):
            exit_rate[k] = exp[k] * 100.0 / n_sample 
            print(f"Exiting Rate of Layer{k}:{exit_rate[k]}")
        acc_correct = acc / n_sample * 100
        print(f"acc_val={acc_correct}")
        return acc_correct
