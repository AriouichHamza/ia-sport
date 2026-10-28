#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from utils.rag_feature_agent_langchain import RAGFeatureAgent

# from modules.Contextualization import Contextualization

from typing import Any
class Contextualization:
    def __init__(self, dataframe, user_prompt: str,context_stack:Any):
        """
        :param dataframe: pandas DataFrame already loaded externally.
        :param user_prompt: description of the user's goal.
        """
        self.df = dataframe
        self.user_prompt = user_prompt
        self.result = {}
        self.context_stack = context_stack
        

    def push_context(
        self,
        user_prompt: str,
        target: str,
        context_global: str,
        use_case: str,
        features: list):
        """Push results into the ContextStack."""
        self.context_stack.user_prompt = user_prompt
        self.context_stack.global_context = context_global
        self.context_stack.data_profiling["target"] = target
        self.context_stack.use_case = use_case
        self.context_stack.data_profiling["features"] = features or []



    def run(self):
        """Run the contextualization pipeline."""
        if self.df is None or self.df.empty:
            raise ValueError("Empty or None dataframe passed to Contextualization.")

        print(f"[✔] DataFrame received: {self.df.shape[0]} rows × {self.df.shape[1]} columns")

        agent = RAGFeatureAgent()
        self.result = agent.analyze(self.df, self.user_prompt)
    


        print(f"[✔] Context detected: {self.result['context_type']} (ID={self.result['context_id']})")
        print(f"[✔] Target column: {self.result['target_column']}")
        print(f"[✔] Use case detected: {self.result['use_case']} ({self.result['use_case_title']})")

        #self.context_stack.push_context(self.user_prompt ,self.result.get('target_column'),self.result.get('context_type'))

        self.push_context(self.user_prompt,
        self.result.get("target_column"),
        self.result.get("context_type"),
        self.result.get("use_case"),
        self.result.get("relevant_features"))

        return self.result

