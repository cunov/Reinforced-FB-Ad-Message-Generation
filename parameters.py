import datetime

class Parameters():
    def __init__(self):
        today = datetime.datetime.now().date()
        base_dir = 'C:/Users/Colton/OneDrive/School/Thesis/Adfenix/play/'
        
        self.MAX_SEQ_LEN = 512
        self.n_hot_masks = 10
        self.n_loc_masks = 2
        self.masking_scheme = 'prioritize_clicks'
        self.device = 'cuda'
        self.loc_score_thresh = 0.75
        self.max_output_length = 70
        self.max_input_length = self.MAX_SEQ_LEN - self.max_output_length - 1
        self.fluency_scaler = 10.0

        self.hotwords_filename = base_dir + 'hotwords.csv'
        self.dataset_filename = base_dir + 'df_masks.csv'

        # These are for pretraining
        self.model_today_dir = base_dir + 'models/{}/'.format(today)

        self.base_bert_dir = self.model_today_dir + 'bert/'
        self.base_coverage_dir = self.model_today_dir + 'coverage/'
        self.base_fluency_dir = self.model_today_dir + 'fluency/'
        self.base_summarizer_dir = self.model_today_dir + 'summarizer/'

        self.bert_tokenizer_dir = self.base_bert_dir + 'tokenizer'
        self.coverage_tokenizer_dir = self.base_coverage_dir + 'tokenizer'
        self.fluency_tokenizer_dir = self.base_fluency_dir + 'tokenizer'
        self.summarizer_tokenizer_dir = self.base_summarizer_dir + 'tokenizer'
        
        self.bert_model_dir = self.base_bert_dir + 'model_{}'
        self.coverage_model_dir = self.base_coverage_dir + 'model_{}'
        self.fluency_model_dir = self.base_fluency_dir + 'model_{}'
        self.summarizer_model_dir = self.base_summarizer_dir + 'model_{}'

        # These are for models with other model dependencies (i.e. summary loop and coverage)
        self.trained_bert_tokenizer_dir = 'C:/Users/Colton/OneDrive/School/Thesis/Adfenix/play/models/2022-03-25/bert/tokenizer'
        self.trained_coverage_tokenizer_dir = 'C:/Users/Colton/OneDrive/School/Thesis/Adfenix/play/models/2022-03-26/coverage/tokenizer'
        self.trained_fluency_tokenizer_dir = 'C:/Users/Colton/OneDrive/School/Thesis/Adfenix/play/models/2022-03-27/fluency_5e_5/tokenizer'
        self.trained_summarizer_tokenizer_dir = 'C:/Users/Colton/OneDrive/School/Thesis/Adfenix/play/models/2022-03-25/summarizer/tokenizer'
        self.trained_bert_model_dir = 'C:/Users/Colton/OneDrive/School/Thesis/Adfenix/play/models/2022-03-25/bert/model_9'
        self.trained_coverage_model_dir = 'C:/Users/Colton/OneDrive/School/Thesis/Adfenix/play/models/2022-03-26/coverage/model_14'
        self.trained_fluency_model_dir = 'C:/Users/Colton/OneDrive/School/Thesis/Adfenix/play/models/2022-03-27/fluency_5e_5/model_29'
        self.trained_summarizer_model_dir = 'C:/Users/Colton/OneDrive/School/Thesis/Adfenix/play/models/2022-03-25/summarizer/model_2'

    def write_params(self, dirname, lines=None):
        params = {}
        params['MAX_SEQ_LEN'] = self.MAX_SEQ_LEN
        params['n_hot_masks'] = self.n_hot_masks
        params['n_loc_masks'] = self.n_loc_masks
        params['masking_scheme'] = self.masking_scheme
        params['locs_score_thresh'] = self.loc_score_thresh
        params['max_output_length'] = self.max_output_length
        params['max_input_length'] = self.max_input_length
        params['fluency_scaler'] = self.fluency_scaler

        if lines is None:
            lines = [key + ' = ' + str(val) + '/n' for key,val in params.items()]
            typ = 'w+'
        else:
            typ = 'a'
        with open(dirname + 'params.txt', typ) as f:
            f.writelines(lines)
            

