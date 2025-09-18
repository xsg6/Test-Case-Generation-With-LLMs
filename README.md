准备数据
数据是 APPS。APPS 有三个难度级别，即入门级（introductory）、面试级（interview）和竞赛级（competition）。
这里是通过app数据集中的不同部分进行的分类。
在处理数据之前，需要下载原始数据集文件。
原链接：https://github.com/hendrycks/apps?tab=readme-ov-file
要处理 APPS 数据，请运行：
plaintext
python preprocess_apps.py [level] [你的APPS根数据路径]
其中 level 可以是 intro、inter 和 comp。这将在../data/文件夹中生成一个.gz 文件。
这里的文件夹自定义即可。
运行 DebateCoder
plaintext
CUDA_VISIBLE_DEVICES=5 python main.py -A claude-3-5-sonnet-20241022 -B gpt-4o-mini  -l python -d appsnew -e 10 -a 0 -b 1 --dataset CodeContest --cur_exp_name claude-4o-simple-test-codecontest-easy --start_idx 0 --end_idx 200  --level easy
其中-A和-B表示参与辩论的模型 A 和模型 B，cur_exp_name是存储结果的文件夹名称。
测试
在运行 DebateCoder 之前，请在/util/config.py中填写你的 API 密钥。
API密钥获取方法：
目前考虑官网获取密钥
要测试 APPS 的结果，假设你的结果位于claude-4o-medium-test，请运行：
plaintext
python calculate_apps.py claude-4o-medium-test 1 100 final claude-3-5-sonnet-20241022 gpt-4o-mini
