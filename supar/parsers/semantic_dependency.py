# -*- coding: utf-8 -*-

import os

import torch
import torch.nn as nn
from supar.models import BiaffineSemanticDependencyModel
from supar.parsers.parser import Parser
from supar.utils import Config, Dataset, Embedding
from supar.utils.common import pad, unk
from supar.utils.field import ChartField, Field, SubwordField
from supar.utils.logging import get_logger, progress_bar
from supar.utils.metric import ChartMetric
from supar.utils.transform import CoNLL

logger = get_logger(__name__)


class BiaffineSemanticDependencyParser(Parser):
    r"""
    The implementation of Biaffine Semantic Dependency Parser.

    References:
        - Timothy Dozat and Christopher D. Manning. 20178.
          `Simpler but More Accurate Semantic Dependency Parsing`_.

    .. _Simpler but More Accurate Semantic Dependency Parsing:
        https://www.aclweb.org/anthology/P18-2077/
    """

    NAME = 'biaffine-semantic-dependency'
    MODEL = BiaffineSemanticDependencyModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.args.feat in ('char', 'bert'):
            self.WORD, self.FEAT = self.transform.FORM
        else:
            self.WORD, self.FEAT = self.transform.FORM, self.transform.CPOS
        self.EDGE, self.LABEL = self.transform.PHEAD

    def train(self, train, dev, test, buckets=32, batch_size=5000, verbose=True, **kwargs):
        r"""
        Args:
            train/dev/test (list[list] or str):
                Filenames of the train/dev/test datasets.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding the unconsumed arguments that can be used to update the configurations for training.
        """


        return super().train(**Config().update(locals()))

    def evaluate(self, data, buckets=8, batch_size=5000, verbose=True, **kwargs):
        r"""
        Args:
            data (str):
                The data for evaluation, both list of instances and filename are allowed.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding the unconsumed arguments that can be used to update the configurations for evaluation.

        Returns:
            The loss scalar and evaluation results.
        """

        return super().evaluate(**Config().update(locals()))

    def predict(self, data, pred=None, buckets=8, batch_size=5000, verbose=True, **kwargs):
        r"""
        Args:
            data (list[list] or str):
                The data for prediction, both a list of instances and filename are allowed.
            pred (str):
                If specified, the predicted results will be saved to the file. Default: ``None``.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            prob (bool):
                If ``True``, outputs the probabilities. Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding the unconsumed arguments that can be used to update the configurations for prediction.

        Returns:
            A :class:`~supar.utils.Dataset` object that stores the predicted results.
        """

        return super().predict(**Config().update(locals()))

    def _train(self, loader):
        self.model.train()

        bar, metric = progress_bar(loader), ChartMetric()

        for words, feats, edges, labels in bar:
            self.optimizer.zero_grad()

            mask = words.ne(self.WORD.pad_index)
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            s_edge, s_label = self.model(words, feats)
            loss = self.model.loss(s_edge, s_label, edges, labels, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            self.optimizer.step()
            self.scheduler.step()

            chart_preds = self.model.decode(s_edge, s_label, mask)
            metric(chart_preds, labels.masked_fill(~(edges.gt(0) & mask), -1))
            bar.set_postfix_str(f"lr: {self.scheduler.get_last_lr()[0]:.4e} - loss: {loss:.4f} - {metric}")

    @torch.no_grad()
    def _evaluate(self, loader):
        self.model.eval()

        total_loss, metric = 0, ChartMetric()

        for words, feats, edges, labels in loader:
            mask = words.ne(self.WORD.pad_index)
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            s_edge, s_label = self.model(words, feats)
            loss = self.model.loss(s_edge, s_label, edges, labels, mask)
            total_loss += loss.item()

            chart_preds = self.model.decode(s_edge, s_label, mask)
            metric(chart_preds, labels.masked_fill(~(edges.gt(0) & mask), -1))
        total_loss /= len(loader)

        return total_loss, metric

    @torch.no_grad()
    def _predict(self, loader):
        self.model.eval()

        preds = {}
        charts, probs = [], []
        for words, feats in progress_bar(loader):
            mask = words.ne(self.WORD.pad_index)
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            lens = mask[:, 0].sum(-1).tolist()
            s_edge, s_label = self.model(words, feats)
            charts.extend(self.model.decode(s_edge, s_label, mask))
            if self.args.prob:
                edge_probs = s_edge.softmax(-1)
                probs.extend([prob[:i, :i].cpu() for i, prob in zip(lens, edge_probs.unbind())])
        charts = [[[self.LABEL.vocab[i] if i >= 0 else None for i in row] for row in chart] for chart in charts]
        preds = {'labels': [CoNLL.build_relations(chart) for chart in charts]}
        if self.args.prob:
            preds['probs'] = probs

        return preds

    @classmethod
    def build(cls, path, min_freq=2, fix_len=20, **kwargs):
        r"""
        Build a brand-new Parser, including initialization of all data fields and model parameters.

        Args:
            path (str):
                The path of the model to be saved.
            min_freq (str):
                The minimum frequency needed to include a token in the vocabulary. Default: 2.
            fix_len (int):
                The max length of all subword pieces. The excess part of each piece will be truncated.
                Required if using CharLSTM/BERT.
                Default: 20.
            kwargs (dict):
                A dict holding the unconsumed arguments.
        """
        args = Config(**locals())
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path) and not args.build:
            # 加载已有模型
            parser = cls.load(**args)
            parser.model = cls.MODEL(**parser.args)
            parser.model.load_pretrained(parser.WORD.embed).to(args.device)
            return parser

        logger.info("Building the fields")
        WORD = Field('words', pad=pad, unk=unk, lower=True)
        if args.feat == 'char':
            FEAT = SubwordField('chars', pad=pad, unk=unk, fix_len=args.fix_len)
        # 怎么用bert，学习一下
        elif args.feat == 'bert':
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.bert)
            FEAT = SubwordField('bert',
                                pad=tokenizer.pad_token,
                                unk=tokenizer.unk_token,
                                fix_len=args.fix_len,
                                tokenize=tokenizer.tokenize)
            FEAT.vocab = tokenizer.get_vocab()
        else:
            FEAT = Field('tags')
        EDGE = ChartField('edges', use_vocab=False, fn=CoNLL.get_edges)
        LABEL = ChartField('labels', fn=CoNLL.get_labels)
        # 对于图的边和标签的抽取，定义了新的field

        if args.feat in ('char', 'bert'):
            transform = CoNLL(FORM=(WORD, FEAT), PHEAD=(EDGE, LABEL))
        else:
            transform = CoNLL(FORM=WORD, POS=FEAT, PHEAD=(EDGE, LABEL))

        train = Dataset(transform, args.train)
        WORD.build(train, args.min_freq, (Embedding.load(args.embed, args.unk) if args.embed else None))
        FEAT.build(train)
        LABEL.build(train)
        args.update({
            'n_words': WORD.vocab.n_init,
            'n_feats': len(FEAT.vocab),
            'n_labels': len(LABEL.vocab),
            'pad_index': WORD.pad_index,
            'unk_index': WORD.unk_index,
            'feat_pad_index': FEAT.pad_index
        })
        model = cls.MODEL(**args)
        model.load_pretrained(WORD.embed).to(args.device)
        return cls(args, model, transform)
