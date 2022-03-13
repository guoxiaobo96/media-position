def label_score_analysis_back(
    misc_args: MiscArgument,
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
    analysis_args: AnalysisArguments,
    ground_truth: str
) -> Dict:
    data_map = BaselineArticleMap()
    bias_distance_matrix = np.zeros(shape=(len(data_map.dataset_bias),len(data_map.dataset_bias)))
    allsides_distance_order_matrix = np.zeros(shape=(len(data_map.dataset_bias),len(data_map.dataset_bias)),dtype=np.int)
    for i,media_a in enumerate(data_map.dataset_list):
        temp_distance = list()
        for j,media_b in enumerate(data_map.dataset_list):
            bias_distance_matrix[i][j] = abs(data_map.dataset_bias[media_a] - data_map.dataset_bias[media_b])
            temp_distance.append(abs(data_map.dataset_bias[media_a] - data_map.dataset_bias[media_b]))
        distance_set = set(temp_distance)
        distance_set = sorted(list(distance_set))
        for o, d_o in enumerate(distance_set):
            for j,d_j in enumerate(temp_distance):
                if d_o == d_j:
                    allsides_distance_order_matrix[i][j] = o
                    
    if ground_truth == 'source':
        distance_base = 'trust'
    else:
        distance_base = 'source'

    baseline_model = joblib.load(
        './log/baseline/model/baseline_'+ground_truth+'_article.c')
    base_model = joblib.load(
        './log/baseline/model/baseline_'+distance_base+'_article.c')
    pew_distance_matrix = np.load('./log/baseline/model/baseline_'+ground_truth+'_article.npy')
    pew_distance_order_matrix = np.zeros(shape=(len(data_map.dataset_bias),len(data_map.dataset_bias)),dtype=np.int)
    for i,media_a in enumerate(data_map.dataset_list):
        temp_distance = pew_distance_matrix[i]
        distance_set = set(temp_distance)
        distance_set = sorted(list(distance_set))
        for o, d_o in enumerate(distance_set):
            for j,d_j in enumerate(temp_distance):
                if d_o == d_j:
                    pew_distance_order_matrix[i][j] = o
        



    analysis_result = dict()
    model_list = dict()
    analysis_data = dict()
    sentence_position_data = dict()

    if not os.path.exists(analysis_args.analysis_result_dir):
        os.makedirs(analysis_args.analysis_result_dir)
    analysis_record_file = '/'.join(analysis_args.analysis_result_dir.split('/')[:6])
    if not os.path.exists(analysis_record_file):
        os.makedirs(analysis_record_file)
    analysis_record_file = os.path.join(analysis_record_file,'record')

    error_count = 0
    analysis_data_temp = get_label_data(misc_args, analysis_args, data_args)
    index = 0
    for k, item in tqdm(analysis_data_temp.items(), desc="Load data"):
        for position, v in item.items():
            if len(v) != len(data_map.dataset_list):
                continue
            try:
                sentence_position_data[index] = {
                    'sentence': k, 'position': position, 'word': k.split(' ')[int(position)]}
                analysis_data[index] = dict()
                for dataset in data_map.dataset_list:
                    analysis_data[index][dataset] = v[dataset]
                index += 1
            except (IndexError, KeyError):
                length = len(k.split(' '))
                error_count += 1
                continue
        if misc_args.global_debug and index > 100:
            break
    analysis_data['media_average'] = dict()

    # analysis_data['concatenate'] = dict()
    # for k, v in analysis_data.items():
    #     for media, item in v.items():
    #         if media not in analysis_data['concatenate']:
    #             analysis_data['concatenate'][media] = dict()
    #         for w, c in item.items():
    #             if w not in analysis_data['concatenate'][media]:
    #                 analysis_data['concatenate'][media][w] = c
    #             else:
    #                 analysis_data['concatenate'][media][w] = float(analysis_data['concatenate'][media][w]) + float(c)

    method = str()
    if analysis_args.analysis_compare_method == 'cluster':
        method = analysis_args.analysis_cluster_method
        analysis_model = ClusterAnalysis(
            misc_args, model_args, data_args, training_args, analysis_args)
        precomputed_analysis_model = ClusterAnalysis(
            misc_args, model_args, data_args, training_args, analysis_args, pre_computer=True)
    elif analysis_args.analysis_compare_method == 'distance':
        method = analysis_args.analysis_distance_method
        analysis_model = DistanceAnalysis(
            misc_args, model_args, data_args, training_args, analysis_args)
    for k, v in tqdm(analysis_data.items(), desc="Build cluster"):
        if k == 'media_average' or k == 'concatenate':
            continue
        try:
            model, cluster_result, dataset_list, encoded_list = analysis_model.analyze(
                v, str(k), analysis_args, keep_result=False)
            analysis_result[k] = cluster_result
            model_list[k] = model
            for i, encoded_data in enumerate(encoded_list):
                if dataset_list[i] not in analysis_data['media_average']:
                    analysis_data['media_average'][dataset_list[i]] = list()
                analysis_data['media_average'][dataset_list[i]].append(
                    encoded_data)
        except ValueError:
            continue
    average_distance_matrix = np.zeros(
        (len(data_map.dataset_list), len(data_map.dataset_list)))

    conclusion = dict()
    # for k, v in analysis_result.items():
    #     analysis_file = os.path.join(analysis_args.analysis_result_dir, k.split('.')[0])
    #     with open(analysis_file, mode='a',encoding='utf8') as fp:
    #         fp.write(json.dumps({'encode': analysis_args.analysis_encode_method,'method':method, 'result':v},ensure_ascii=False)+'\n')
    #     for country, distance in v.items():
    #         if country not in conclusion:
    #             conclusion[country] = dict()
    #         conclusion[country][k] = distance

    if analysis_args.analysis_compare_method == 'distance':
        for k, v in analysis_result.items():
            label_list, data = v
            _draw_heatmap(data, label_list, label_list)
            plt_file = os.path.join(analysis_args.analysis_result_dir,
                                    analysis_args.analysis_encode_method+'_'+method+'_' + k.split('.')[0]+'.png')
            plt.savefig(plt_file, bbox_inches='tight')
            plt.close()
    else:
        model_list['base'] = baseline_model
        model_list['distance_base'] = base_model
        cluster_compare = ClusterCompare(misc_args, analysis_args)
        analysis_result = cluster_compare.compare(model_list)

        for i, dataset_name_a in enumerate(tqdm(data_map.dataset_list, desc="Combine cluster")):
            for j, dataset_name_b in enumerate(data_map.dataset_list):
                if i == j or average_distance_matrix[i][j] != 0:
                    continue
                average_distance = 0
                encoded_a = analysis_data['media_average'][dataset_name_a]
                encoded_b = analysis_data['media_average'][dataset_name_b]
                for k in range(len(encoded_a)):
                    if k in analysis_result and (analysis_result[k] < analysis_args.analysis_threshold or analysis_args.analysis_threshold == -1):
                        # average_distance += cosine_distances(
                        #     encoded_a[k].reshape(1, -1), encoded_b[k].reshape(1, -1))[0][0]
                        average_distance += euclidean_distances(
                            encoded_a[k].reshape(1, -1), encoded_b[k].reshape(1, -1))[0][0]
                average_distance_matrix[i][j] = average_distance / \
                    len(encoded_a)
                average_distance_matrix[j][i] = average_distance / \
                    len(encoded_a)
        analysis_data['media_average'] = average_distance_matrix

        model, cluster_result, _, _ = precomputed_analysis_model.analyze(
            analysis_data['media_average'], 'media_average', analysis_args, encode=False, dataset_list=list(data_map.dataset_list))
        model_list['media_average'] = model

        cluster_average = list()
        for _, v in analysis_result.items():
            if v < analysis_args.analysis_threshold or analysis_args.analysis_threshold == -1:
                cluster_average.append(v)

        analysis_result = cluster_compare.compare(model_list)
        analysis_result['cluster_average'] = np.mean(cluster_average)
        analysis_result = sorted(analysis_result.items(), key=lambda x: x[1])
        sentence_position_data['media_average'] = {
            'sentence': 'media_average', 'position': -2, 'word': 'media_average'}
        sentence_position_data['cluster_average'] = {
            'sentence': 'cluster_average', 'position': -2, 'word': 'cluster_average'}
        sentence_position_data['distance_base'] = {
            'sentence': 'distance_base', 'position': -2, 'word': 'distance_base'}

        media_distance = analysis_data["media_average"]
        media_distance_order_matrix = np.zeros(shape=(len(data_map.dataset_bias),len(data_map.dataset_bias)),dtype=np.int)

        for i,media_a in enumerate(data_map.dataset_list):
            temp_distance = list()
            for j,media_b in enumerate(data_map.dataset_list):
                temp_distance.append(media_distance[i][j])
            order_list = np.argsort(temp_distance)
            order_list = order_list.tolist()
            for j in range(len(data_map.dataset_list)):
                order = order_list.index(j)
                media_distance_order_matrix[i][j] = order

        allsides_rank_similarity = 0
        for i in range(len(data_map.dataset_list)):
            # sort_distance += euclidean_distances(media_distance_order_matrix[i].reshape(1,-1), distance_order_matrix[i].reshape(1,-1))
            tau, p_value = kendalltau(media_distance_order_matrix[i].reshape(1,-1), allsides_distance_order_matrix[i].reshape(1,-1))
            allsides_rank_similarity += tau
        allsides_rank_similarity /= len(data_map.dataset_list)

        pew_rank_similarity = 0
        for i in range(len(data_map.dataset_list)):
            # sort_distance += euclidean_distances(media_distance_order_matrix[i].reshape(1,-1), distance_order_matrix[i].reshape(1,-1))
            tau, p_value = kendalltau(media_distance_order_matrix[i].reshape(1,-1), pew_distance_order_matrix[i].reshape(1,-1))
            pew_rank_similarity += tau
        pew_rank_similarity /= len(data_map.dataset_list)

        pew_cosine_similarity = 0
        for i,media_a in enumerate(data_map.dataset_list):
            distance = cosine_similarity(pew_distance_matrix[i].reshape(1,-1),media_distance[i].reshape(1,-1))
            pew_cosine_similarity += distance[0][0]
        pew_cosine_similarity /= len(data_map.dataset_list)




        result = dict()
        average_distance = dict()
        for k, v in tqdm(analysis_result, desc="Combine cluster analyze"):
            sentence = sentence_position_data[k]['sentence']
            position = sentence_position_data[k]['position']
            word = sentence_position_data[k]['word']
            if sentence not in result:
                average_distance[sentence] = list()
                result[sentence] = dict()
            result[sentence][position] = (v, word)
            average_distance[sentence].append(v)

        for sentence, average_distance in average_distance.items():
            result[sentence][-1] = (np.mean(average_distance),
                                    'sentence_average')

        sentence_list = list(result.keys())
        analysis_result = {k: {'score': v, 'sentence': sentence_list.index(
            sentence_position_data[k]['sentence'])+1, 'position': sentence_position_data[k]['position'], 'word': sentence_position_data[k]['word']} for k, v in analysis_result}

        result_path = os.path.join(
            analysis_args.analysis_result_dir, analysis_args.graph_distance)
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        result_file = os.path.join(result_path, analysis_args.analysis_encode_method +
                                   '_'+method+'_'+analysis_args.graph_kernel+'_sort_'+ground_truth+'.json')
        with open(result_file, mode='w', encoding='utf8') as fp:
            for k, v in analysis_result.items():
                fp.write(json.dumps(v, ensure_ascii=False)+'\n')

        result_file = os.path.join(result_path, analysis_args.analysis_encode_method +
                                   '_'+method+'_'+analysis_args.graph_kernel+'_sentence_'+ground_truth+'.json')
        with open(result_file, mode='w', encoding='utf8') as fp:
            for k, v in result.items():
                v['sentence'] = k
                fp.write(json.dumps(v, ensure_ascii=False)+'\n')

        record_item = {'baseline':ground_truth,'augmentation_method':data_args.data_type.split('/')[0],'cluster_performance':round(result['media_average'][-2][0],2),'allsides_rank_similarity':round(allsides_rank_similarity,2),'pew_rank_similarity':round(pew_rank_similarity,2),'pew_cosine_similarity':round(pew_cosine_similarity,2)}
        with open(analysis_record_file,mode='a',encoding='utf8') as fp:
            fp.write(json.dumps(record_item,ensure_ascii=False)+'\n')
    # print("The basic distance is {}".format(result['distance_base'][-2][0]))
    print("The allsides rank similarity is {}".format(round(allsides_rank_similarity,2)))
    print("The pew rank similarity is {}".format(round(pew_rank_similarity,2)))
    print("The pew cosine similarity is {}".format(round(pew_cosine_similarity,2)))
    print("The media average performance is {}".format(
        result['media_average'][-2][0]))

    print("Analysis finish")
    return analysis_result
