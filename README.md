# predictability_tipping
実行時のディレクトリに注意が必要であり、基本的に実行は./tipping_elementにて行う

実行時には必ずカレントディレクトリに標準出力を書き込むoutput.logという名前のlogファイルが生成されるようにする必要がある（$nohup python3 ~.py >output.log &など)

あらかじめ、このディレクトリの親ディレクトリにoutput/amazon, output/amocの二つのディレクトリを生成しておく
