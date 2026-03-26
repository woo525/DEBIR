import os
import pickle
import glob


if __name__ == "__main__":

    root_path = "./warm_up_labels/NUNI_US64"

    t_flag_list = ['train','test']
    for  t_flag in t_flag_list:
        psnr_paths = glob.glob(os.path.join(root_path, "psnrs_*_%s.txt") %(t_flag))
        
        total_psnr = {}

        for psnr_path in psnr_paths:

            data_dict = {}
            with open(psnr_path, "r") as f:
                for line in f:
                    key, value = line.strip().split(": ")

                    name = key.split('_tensor')[0]
                    k = key.split('(')[1].split(',')[0]
                    hi = key.split('(')[2].split(',')[0]    
                    wi = key.split('(')[3].split(',')[0]

                    new_key = name +'/'+ k +'/'+ hi +'/'+ wi

                    data_dict[new_key] = float(value)

            sorted_dict = dict(sorted(data_dict.items()))
            psnr_name = psnr_path.split('/')[-1].split('.')[0]
            total_psnr[psnr_name] = sorted_dict

        exp_types = list(total_psnr.keys())
        best_per_scene = {}
        
        max_psnr_sum = 0
        
        for i in range(len(total_psnr[exp_types[0]])):
            max_psnr = 0
            max_exp_type = 'None'
            max_name = 'None'

            flag = list(total_psnr[exp_types[0]].keys())[i]
            # print(flag)
            for exp_type in exp_types:

                if max_psnr < total_psnr[exp_type][list(total_psnr[exp_type].keys())[i]]:
                    if flag != list(total_psnr[exp_type].keys())[i]:
                        import pdb;pdb.set_trace() ##
                    max_psnr = total_psnr[exp_type][list(total_psnr[exp_type].keys())[i]]
                    max_exp_type = exp_type
                    max_name = list(total_psnr[exp_type].keys())[i]

            best_per_scene[max_name] = max_exp_type.split('s_')[-1].split('_t')[0]
        
        out_path = os.path.join(root_path, "NUNI_US64_%s.txt" %(t_flag))
        with open(out_path, "w") as f:
            for key, value in best_per_scene.items():
                f.write(f"{key}: {value}\n")    
