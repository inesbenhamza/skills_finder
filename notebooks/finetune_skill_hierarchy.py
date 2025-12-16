#!/usr/bin/env python3
"""
Fine-tune skill embeddings to preserve hierarchical relationships.

This script:
1. Parses a hierarchical skill tree
2. Creates positive pairs (parent-child relationships)
3. Fine-tunes embeddings using contrastive learning
4. Ensures parent skills are close to their child skills in embedding space
"""

import os
import sys
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import re
import torch
import pandas as pd
import yaml


os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
torch.set_default_device('cpu')

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def parse_skill_tree(tree_text: str) -> Dict[str, List[str]]:

    hierarchy = {}
    lines = tree_text.strip().split('\n')
    stack = []  
    
    for line in lines:
        if not line.strip():
            continue
        
        
        indent = len(line) - len(line.lstrip())
        skill_name = line.strip().lower()
        
        
        while stack and stack[-1][0] >= indent:
            stack.pop()
        
        
        if stack:
            parent_path = stack[-1][1]
            full_path = f"{parent_path} {skill_name}".strip()
        else:
            full_path = skill_name
        
        if stack:
            parent = stack[-1][1]
            if parent not in hierarchy:
                hierarchy[parent] = []
            hierarchy[parent].append(full_path)
        else:
           
            if full_path not in hierarchy:
                hierarchy[full_path] = []
        
        stack.append((indent, full_path))
    
    return hierarchy


def build_skill_pairs(hierarchy: Dict[str, List[str]]) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Build positive and negative pairs from hierarchy.
    
    Positive pairs: (parent, child) - should be close in embedding space
    Negative pairs: (unrelated skills) - should be far in embedding space
    
    Returns:
        (positive_pairs, negative_pairs)
    """
    positive_pairs = []
    negative_pairs = []
    
   
    all_skills = set()
    for parent, children in hierarchy.items():
        all_skills.add(parent)
        for child in children:
            all_skills.add(child)
            # Recursively get children of children
            if child in hierarchy:
                for grandchild in hierarchy[child]:
                    all_skills.add(grandchild)
    
    all_skills = sorted(list(all_skills))
    
    # Create positive pairs: parent -> child
    for parent, children in hierarchy.items():
        for child in children:
            
            positive_pairs.append((parent, child))
            
            # between parent and grandchildren
            if child in hierarchy:
                for grandchild in hierarchy[child]:
                    positive_pairs.append((parent, grandchild))
                    positive_pairs.append((child, grandchild))
            
            # Sibling relationships (children of same parent should be somewhat close)
            for sibling in children:
                if sibling != child:
                    positive_pairs.append((child, sibling))
    
    # Create negative pairs: unrelated skills
    # FIXED: Increased ratio and ensure all root-level skills are negative pairs
    skills_list = list(all_skills)
    
    
    root_skills = []
    all_children = set()
    for parent, children in hierarchy.items():
        all_children.update(children)
        # Recursively get all descendants
        for child in children:
            if child in hierarchy:
                all_children.update(hierarchy[child])
    
    for skill in skills_list:
        if skill not in all_children:
            root_skills.append(skill)
    
    # Ensure ALL root-level skills are negative pairs
    print(f"  Found {len(root_skills)} root-level skills")
    for i, skill1 in enumerate(root_skills):
        for skill2 in root_skills[i+1:]:
            # Check if they're related in hierarchy
            are_related = False
            for parent, children in hierarchy.items():
                if (skill1 == parent and skill2 in children) or \
                   (skill2 == parent and skill1 in children):
                    are_related = True
                    break
            if not are_related:
                negative_pairs.append((skill1, skill2))
    
    
    max_negatives = len(positive_pairs) * 5  
    
    for i, skill1 in enumerate(skills_list):
        if len(negative_pairs) >= max_negatives:
            break
        for j, skill2 in enumerate(skills_list[i+1:], start=i+1):
            if len(negative_pairs) >= max_negatives:
                break
            
            # Skip if already added as root-level negative pair
            if (skill1, skill2) in negative_pairs or (skill2, skill1) in negative_pairs:
                continue
            
            # Check if they're related (same parent or parent-child)
            are_related = False
            for parent, children in hierarchy.items():
                if (skill1 == parent and skill2 in children) or \
                   (skill2 == parent and skill1 in children) or \
                   (skill1 in children and skill2 in children):
                    are_related = True
                    break
                # Check if one is ancestor of the other
                if skill1 in hierarchy:
                    if skill2 in hierarchy[skill1] or any(skill2 in hierarchy.get(gc, []) for gc in hierarchy[skill1]):
                        are_related = True
                        break
                if skill2 in hierarchy:
                    if skill1 in hierarchy[skill2] or any(skill1 in hierarchy.get(gc, []) for gc in hierarchy[skill2]):
                        are_related = True
                        break
            
            if not are_related:
                # Check if they share common words (might still be related)
                words1 = set(skill1.split())
                words2 = set(skill2.split())
                if not words1.intersection(words2):  # No common words
                    negative_pairs.append((skill1, skill2))
    
    return positive_pairs, negative_pairs


def normalize_skill_name(skill: str) -> str:
    """
    Normalize skill name to match dataset format.
    Converts "microsoft power bi" -> "Microsoft Power BI"
    """
    # Capitalize first letter of each word
    words = skill.split()
    normalized = ' '.join(word.capitalize() for word in words)
    
    # Handle common acronyms (preserve uppercase)
    acronyms = {
        'bi': 'BI',
        'jira': 'JIRA',
        'sql': 'SQL',
        'ssrs': 'SSRS',
        'ssis': 'SSIS',
        'ssas': 'SSAS',
        'aws': 'AWS',
        'api': 'API',
        'etl': 'ETL',
        'cpm': 'CPM',
        'epm': 'EPM',
        'odi': 'ODI',
        'hfm': 'HFM',
        'bpc': 'BPC',
        'idq': 'IDQ',
        'mdx': 'MDX',
        'dax': 'DAX',
        't-sql': 'T-SQL',
        'pl/sql': 'PL/SQL',
    }
    
    # Replace acronyms
    for acronym, replacement in acronyms.items():
        normalized = normalized.replace(acronym.capitalize(), replacement)
        normalized = normalized.replace(acronym, replacement)
    
    return normalized


def create_training_examples(positive_pairs: List[Tuple[str, str]], 
                            negative_pairs: List[Tuple[str, str]]) -> List[InputExample]:
    """
    Create InputExample objects for training.
    """
    examples = []
    
    # Positive 
    for text1, text2 in positive_pairs:
        # Normalize skill names
        norm1 = normalize_skill_name(text1)
        norm2 = normalize_skill_name(text2)
        examples.append(InputExample(texts=[norm1, norm2], label=1.0))
    
    # Negative 
    for text1, text2 in negative_pairs:
        norm1 = normalize_skill_name(text1)
        norm2 = normalize_skill_name(text2)
        examples.append(InputExample(texts=[norm1, norm2], label=0.0))
    
    return examples


def build_hierarchy_from_dataset(skills_csv: str) -> Dict[str, List[str]]:
    """
    Automatically build hierarchy from actual skills in the dataset.
    
    Strategy:
    1. Extract all unique skills
    2. Infer parent-child relationships by analyzing skill names:
       - Skills starting with a company name are children (e.g., "Microsoft Power BI" -> parent: "Microsoft")
       - Skills with common prefixes are grouped
    3. Use domain information if available
    
    Args:
        skills_csv: Path to skills CSV file
        
    Returns:
        Dict mapping parent skill -> list of child skills
    """
    print(f"Loading skills from: {skills_csv}")
    df = pd.read_csv(skills_csv)
    
    # Get all unique skills
    unique_skills = sorted(df['SKILLS_DSC'].unique())
    print(f"Found {len(unique_skills)} unique skills")
    
    hierarchy = {}
    
    # Common company
    company_prefixes = [
        'microsoft', 'oracle', 'sap', 'ibm', 'amazon', 'google', 'atlassian',
        'salesforce', 'tableau', 'qlik', 'informatica', 'talend', 'alteryx',
        'databricks', 'snowflake', 'teradata', 'mongodb', 'neo4j'
    ]
    
    # Build hierarchy based on prefixes
    for skill in unique_skills:
        skill_lower = skill.lower()
        words = skill_lower.split()
        
        # Check if skill starts with a known company prefix
        for prefix in company_prefixes:
            if skill_lower.startswith(prefix):
                parent = prefix.capitalize()
                if parent not in hierarchy:
                    hierarchy[parent] = []
                if skill_lower != prefix:
                    if skill not in hierarchy[parent]:
                        hierarchy[parent].append(skill)
                break
        
        # Also check for multi-word patterns (e.g., "Microsoft SQL Server" -> parent: "Microsoft")
        if len(words) > 1:
            first_word = words[0].capitalize()
            # If first word is a known company, add as child
            if first_word.lower() in company_prefixes:
                if first_word not in hierarchy:
                    hierarchy[first_word] = []
                if skill not in hierarchy[first_word]:
                    hierarchy[first_word].append(skill)
    
    # group skills by domain if available
    if 'DOMAIN_DSC' in df.columns:
        for domain in df['DOMAIN_DSC'].unique():
            domain_skills = df[df['DOMAIN_DSC'] == domain]['SKILLS_DSC'].unique()
            if len(domain_skills) > 1:
                domain_key = f"Domain: {domain}"
                if domain_key not in hierarchy:
                    hierarchy[domain_key] = []
                for skill in domain_skills:
                    if skill not in hierarchy[domain_key]:
                        hierarchy[domain_key].append(skill)
    
    print(f"Built hierarchy with {len(hierarchy)} parent nodes")
    return hierarchy


# Skill tree from user's input (fallback if not using dataset)
SKILL_TREE = """
airbyte

alteryx
    analytics hub
    designer

amazon
    aws
      cloudformation
      glue
    quicksight
    redshift
    web services

anaconda enterprise

anaplan for finance

applied olap dodeca

atlassian
  bitbucket
  jira software

azure purview

cch tagetik cpm
      conso
      planning

cloudera
  data
    platform
    warehouse
  enterprise data hub

colibra

data galaxy

databricks
  lakehouse platform

dataiku

dax

dbt

denodo platform

docker

domo

fivetran

fluence

github enterprise

gitlab
  ci

google
  bigquery
  cloud
    dataproc
    platform
  data studio
  vertex ai

hadoop

ibm
  cognos
    analytics
    tm1
  datastage
  db2
  netezza performance server
  spss statistics

informatica
  data quality
      idq
  intelligent cloud services
  powercenter

insightsoftware longview planning

java

javascript

knime analytics platform

kubernetes

looker

matillion etl

matlab

mdx

microsoft
  analytics platform system
        ssas
  azure
    data
      factory
      lake store
    devops services
    machine learning
    sql database
    synapse analytics
  fabric
  power bi
  sql server
      integration services
          ssis                                               
      reporting services
          ssrs

microstrategy

mongodb

neo4j graph database

onestream
  conso
  planning

oracle
  analytics cloud
  business intelligence enterprise edition
  cloud epm
  data integrator
      odi
  database
  financial analytics
  hyperion
    essbase
    financial management
        hfm
    planning
  mysql

pentaho business analytics

pl/sql

postgresql

project management

prophix
  conso
  planning

python

qlik
  sense
  view

r

rapidminer studio

rstudio team

salesforce einstein analytics

sap
  business planning and consolidation
          sap
            bpc
  businessobjects bi
  bw
  bw/4hana
  crystal reports
  data services
  disclosure management
  financial
    consolidation
    information management
  hana
  process control
  profitability and cost management

sas
  base
  enterprise guide

scala

snaplogic intelligent integration platform

snowflake data cloud

spark

sql

t-sql

tableau desktop and online

talend
  data fabric
  open studio
      for data quality
  platform for big data integration

teradata
  database
  vantage

terraform

tibco data science

vba

vena
  close
  solutions budgeting, planning & forecasting

workday adaptive planning

workiva
"""


def main():
    """Main fine-tuning function."""
    print(" FINE-TUNING SKILL EMBEDDINGS FOR HIERARCHICAL RELATIONSHIPS")

    
    
    config_path = os.path.join(project_root, "config", "search_config.yaml")
    use_dataset = True  # set it to False to use hardcoded SKILL_TREE
    
    if use_dataset:
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            skills_csv = config.get("skills_csv", "skills/skillscleaned.csv")
            if not os.path.isabs(skills_csv):
                skills_csv = os.path.join(project_root, skills_csv)
            
            print("\n[1/5] Building hierarchy from actual dataset...")
            hierarchy = build_hierarchy_from_dataset(skills_csv)
        except Exception as e:
            print(f"Could not load from dataset: {e}")
            print("   Falling back to hardcoded SKILL_TREE...")
            use_dataset = False
    
    if not use_dataset:
        hierarchy = parse_skill_tree(SKILL_TREE)
    

    #debug
    print(f"Found {len(hierarchy)} parent skills")
    total_children = sum(len(children) for children in hierarchy.values())
    print(f"Found {total_children} child skills")
    
    #  Build training pairs
    positive_pairs, negative_pairs = build_skill_pairs(hierarchy)
    print(f"✓ Created {len(positive_pairs)} positive pairs (parent-child relationships)")
    print(f"✓ Created {len(negative_pairs)} negative pairs (unrelated skills)")
    
    # Create training examples
    print("\n[3/5] Creating training examples...")
    train_examples = create_training_examples(positive_pairs, negative_pairs)
    print(f"✓ Created {len(train_examples)} training examples")
    
    print("Sample positive pairs:")
    for i, (text1, text2) in enumerate(positive_pairs[:5]):
        print(f"   {i+1}. '{normalize_skill_name(text1)}' <-> '{normalize_skill_name(text2)}'")
    
    print("Sample negative pairs:")
    for i, (text1, text2) in enumerate(negative_pairs[:5]):
        print(f"   {i+1}. '{normalize_skill_name(text1)}' <-> '{normalize_skill_name(text2)}'")
    
    #  Load base model
    base_model = "BAAI/bge-m3"  # Same as config
    model = SentenceTransformer(base_model)
    model = model.to('cpu')
    print(f"✓ Loaded: {base_model} (on CPU)")
    
    
    #fine-tune
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    
    train_loss = losses.ContrastiveLoss(
        model,
        distance_metric=losses.SiameseDistanceMetric.COSINE_DISTANCE,
        margin=1.0 
    )
    
    print(f"Training with {len(train_examples)} examples")
    print(f"Positive pairs: {len(positive_pairs)}, Negative pairs: {len(negative_pairs)}")
    print(f"Using ContrastiveLoss with margin=0.5")
    print("This will make parent-child skills close together in embedding space!")
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=5,
        warmup_steps=100,
        optimizer_params={'lr': 2e-5},
        show_progress_bar=True,
        use_amp=False 
    )
    
    
    output_path = "skills_finetuned_hierarchy"
    model.save(output_path)



if __name__ == "__main__":
    main()

