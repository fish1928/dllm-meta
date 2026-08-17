import re
import os

file_target = 'logs/run_e2e_gsm8k.log'
with open(file_target, 'r') as file:
    lines = [line for line in file.readlines() if not re.search(r'\d{1,3}\.\d{1,2}s\/it', line)]
    lines = [line for line in lines if not re.search(r'\d+\.\d{1,2}it\/s', line)]
    lines = [line for line in lines if not re.search(r'\d+\.\d{1,2} ?examples\/s', line)]
    lines = [line for line in lines if re.search(r'[a-zA-Z0-9]', line)]

    passages = [passage for passage in re.split(r'=== .*? ===\n', ''.join(lines)) if len(passage) > 10]
# end

for passage in passages:
    score = float(re.search(r'\|flexible-extract\| *?5\|exact_match\|↑ *?\|(\d+\.\d+)\|', passage)[1])
    str_setting_all = re.search(r'routers_e2e\/(.*?).pt', passage)[1]
    list_str_setting = str_setting_all.split('__')
    info_run = {'score': score}
    for str_setting in list_str_setting:
        k,v = str_setting.split('-')
        info_run[k] = v
    # end

    print(info_run)
# end