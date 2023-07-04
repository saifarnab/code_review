import copy
import re

import enchant
import nltk
import pandas as pd
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('words')
nltk.download('omw-1.4')
nltk.download('stopwords')


class _DataFrame:
    def __init__(self):
        self.path = ''
        self.regex = True
        self.inplace = True
        self.error_bad_lines = False
        self.dataframe_headers = ['operation', 'scope', 'remove_code', 'add_code', 'message']
        self.wd = None
        self.original_message_dataframe = None
        self.lemmatizer = WordNetLemmatizer()
        self.valid_word_checker = enchant.Dict("en_US")
        self.total_words_list = []
        self.unique_words_set = set()
        self.keywords_set = set()
        self.modified_words_set = set()

    def _get_workable_df(self):
        df = pd.read_csv(self.path)
        wd = df[self.dataframe_headers]  # taking necessary data series
        wd.drop_duplicates(inplace=True)  # drop duplicates 2087*5 to 1636*5
        self.wd = wd[pd.notnull(wd['message'])]  # drop rows if message is nan 1636*5 to 1621*5

    def _modify_operation(self):
        operation_values = [0, 1, 2, 3]  # 1:not_enough, 2:replace, 3:delete, 4:insert, 0:rest of them
        self.wd['operation'].replace({
            'not_enough': 3,
            'replace': 0,
            'delete': 1,
            'insert': 2,
        }, regex=self.regex, inplace=self.inplace)

        for i in self.wd['operation']:
            if i not in operation_values:
                self.wd['operation'].replace(i, 0, inplace=self.inplace)

        self.wd['operation'] = self.wd['operation'].astype(int)

    def _modify_scope(self):
        scope_values = [0, 1]  # already given, 0:token level, 1:line level, 5; rest of them

        self.wd['scope'].replace({
            '0': 0,
            '1': 1,
        }, regex=self.regex, inplace=self.inplace)

        for i in self.wd['scope']:
            if i not in scope_values:
                self.wd['scope'].replace(i, 5, inplace=self.inplace)

        self.wd['scope'] = self.wd['scope'].astype(int)

    def _modify_add_code(self):
        self.wd['add_code'] = self.wd['add_code'].fillna(0)
        for i in self.wd['add_code']:
            if i != 0:
                self.wd['add_code'].replace(i, 1, inplace=True)

    def _modify_remove_code(self):
        self.wd['remove_code'] = self.wd['remove_code'].fillna(0)
        for i in self.wd['remove_code']:
            if i != 0:
                self.wd['remove_code'].replace(i, 1, inplace=True)

    def _review_prepossessing(self):
        self.original_message_dataframe = copy.deepcopy(self.wd)
        for index, i in self.wd.iterrows():
            message = i['message'].replace('\n', ' ')
            i_split = message.split(' ')
            new_string = ""
            for x in i_split:
                x = self.lemmatizer.lemmatize(x)
                self.total_words_list.append(x)
                convert = x.strip()

                if ("(" in x) and (")" in x):
                    convert = "keywordfunction"

                if ".h" in x or "#" in x:
                    convert = "keyworddoth"

                if "_" in x:
                    convert = "keywordunderscore"

                convert = re.sub(r'[^\w\s]', '', convert)

                if any(char.isdigit() for char in convert):
                    convert = "keywordnumeric"

                if convert.strip():
                    convert = str(convert[0].lower() + convert[1:])
                    if any(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" for c in convert):
                        convert = "keywordvariable"

                    if not self.valid_word_checker.check(convert.strip()):
                        self.keywords_set.add(convert.strip())

                    if self.valid_word_checker.check(convert.strip()):
                        if convert in self.modified_words_set:
                            pass
                        elif len(wn.synsets(convert)) > 0:
                            sysnet_object = wn.synsets(convert)[0].name()
                            sysnet_list = wn.synset(sysnet_object).lemma_names()
                            val = None
                            try:
                                val = next(s for s in sysnet_list if s in self.modified_words_set)
                            except:
                                pass
                            if val is None:
                                self.modified_words_set.add(convert)
                            else:
                                convert = val
                    word = convert
                    if word == 'an':
                        word = 'a'
                    if word not in self.unique_words_set:
                        synonyms = []
                        for syn in wn.synsets(word):
                            for l in syn.lemmas():
                                synonyms.append(l.name())
                        v = list(self.unique_words_set & set(synonyms))
                        if v:
                            word = v[0]
                        else:
                            self.unique_words_set.add(word)

                    self.unique_words_set.add(word)
                    new_string += word + ' '

            self.wd['message'][index] = new_string.strip()

    def _drop_irrelevant_data(self):
        dataframe_with_not_enough = self.wd
        operation_value_not_enough_index_list = self.wd.index[self.wd['operation'] == 3].tolist()
        workable_dataframe = self.wd.drop(operation_value_not_enough_index_list)
        operation_value_not_enough_index_list2 = self.original_message_dataframe.index[
            self.original_message_dataframe['operation'] == 3].tolist()
        original_message_dataframe = self.original_message_dataframe.drop(operation_value_not_enough_index_list2)
        return workable_dataframe, original_message_dataframe, dataframe_with_not_enough

    def process(self, path: str):
        self.path = path
        self._get_workable_df()
        self._modify_operation()
        self._modify_scope()
        self._modify_add_code()
        self._modify_remove_code()
        self._review_prepossessing()
        return self._drop_irrelevant_data()


prepossessing = _DataFrame()
