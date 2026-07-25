---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:3340
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
widget:
- source_sentence: maxwell equations
  sentences:
  - dampak kegiatan manusia terhadap komunitas tumbuhan di kalimantan timur kumpulan
    makalah seminar dan penelitian community research
  - sri kresna barata 1 javanese literature
  - solusi persoalan medan magnetik elemen hingga suatu jawaban pintas persamaan maxwell
    kedua maxwell equations
- source_sentence: fiction research
  sentences:
  - religiusitas dalam tiga novel modern kemarau khotbah di atas bukit dan kubah fiction
    research
  - the communist experiment revolution socialism and global conflict in the twentieth
    century communist industrial revolutions
  - the tree of life a book depicting the life of charles darwin naturalist geologist
    thinker darwin charles biography naturalist geologist and thinker
- source_sentence: ederly old age
  sentences:
  - dasar dasar dan praktek irigasi irrigation
  - the economic livelihood of the aged a case study in a village in yogyakarta special
    territory indonesia ederly old age
  - tata gereja gereja protestan di indonesia suatu sumbangan pikiran mengenai sejarah
    dan asas asasnya church
- source_sentence: railroad transportation vietnam
  sentences:
  - from stone age to early civilisation in malaysia empowering identity of race 1
    archaeology and history malaysia 2 antiquities prehistoric malaysia 3 civilization
    malaysia history
  - we have no dreaming national characteristics australian social psychology australia
  - socialist republic of vietnam preparing the ho chi minh city metro rail system
    project financed by japan special fund railroad transportation vietnam
- source_sentence: american drama
  sentences:
  - champion 101 tip motivasi inspirasi sukses menjadi juara sejati success in business
  - best american plays sixth series 1963 1967 american drama
  - mathematics and optimal form nature aesthetics form philosophy motion calculus
    of variations mathematics
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for retrieval.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) <!-- at revision e8f8c211226b894fcb81acc59f3b34ba3efd5f42 -->
- **Maximum Sequence Length:** 128 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'transformer_task': 'feature-extraction', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'last_hidden_state'}}, 'module_output_name': 'token_embeddings', 'architecture': 'BertModel'})
  (1): Pooling({'embedding_dimension': 384, 'pooling_mode': 'mean', 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```
Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'american drama',
    'best american plays sixth series 1963 1967 american drama',
    'mathematics and optimal form nature aesthetics form philosophy motion calculus of variations mathematics',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[ 1.0000,  0.6461, -0.0083],
#         [ 0.6461,  1.0000,  0.0627],
#         [-0.0083,  0.0627,  1.0000]])
```
<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 3,340 training samples
* Columns: <code>anchor</code> and <code>positive</code>
* Approximate statistics based on the first 100 samples:
  |          | anchor                                                                           | positive                                                                          |
  |:---------|:---------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|
  | type     | string                                                                           | string                                                                            |
  | modality | text                                                                             | text                                                                              |
  | details  | <ul><li>min: 3 tokens</li><li>mean: 8.15 tokens</li><li>max: 22 tokens</li></ul> | <ul><li>min: 8 tokens</li><li>mean: 22.88 tokens</li><li>max: 51 tokens</li></ul> |
* Samples:
  | anchor                                               | positive                                                                                                                    |
  |:-----------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------|
  | <code>human psychology</code>                        | <code>fisiologi manusia dari sel ke sistem ed 2 human psychology</code>                                                     |
  | <code>english language grammar</code>                | <code>a grammar of late modern english for the use of continental especially dutch students english language grammar</code> |
  | <code>world history war environmental aspects</code> | <code>world war ii memorial world history war environmental aspects</code>                                                  |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false,
      "directions": [
          "query_to_doc"
      ],
      "partition_mode": "joint",
      "hardness_mode": null,
      "hardness_strength": 0.0
  }
  ```

### Evaluation Dataset

#### Unnamed Dataset

* Size: 716 evaluation samples
* Columns: <code>anchor</code> and <code>positive</code>
* Approximate statistics based on the first 100 samples:
  |          | anchor                                                                           | positive                                                                          |
  |:---------|:---------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|
  | type     | string                                                                           | string                                                                            |
  | modality | text                                                                             | text                                                                              |
  | details  | <ul><li>min: 3 tokens</li><li>mean: 8.97 tokens</li><li>max: 33 tokens</li></ul> | <ul><li>min: 6 tokens</li><li>mean: 25.16 tokens</li><li>max: 75 tokens</li></ul> |
* Samples:
  | anchor                                                                                                                                      | positive                                                                                                                                                                                                                                                                                          |
  |:--------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>indonesia history</code>                                                                                                              | <code>sejarah ringkas perjuangan pasukan tni batalyon iv sektor ii dmtt stt sub teritorium vii komandan sumatera indonesia history</code>                                                                                                                                                         |
  | <code>journalist in women gender effects on identity and relation by indonesian female journalis forum at mas media electronic medan</code> | <code>identitas dan relasi gender jurnalis perempuan studi kasus jurnalis perempuan yang begabung di fjpi dan bekerja dimedia cetak dan elektronik di medan journalist in women gender effects on identity and relation by indonesian female journalis forum at mas media electronic medan</code> |
  | <code>botanical chemistry</code>                                                                                                            | <code>isolasi pemisahan dan pemurnian senyawa berkhasiat yang terkandung pada daun tumbuhan sempil pil dicranopteris dichotoma yang digunakan sebagai obatpenyakit kanker secara tradisional botanical chemistry</code>                                                                           |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false,
      "directions": [
          "query_to_doc"
      ],
      "partition_mode": "joint",
      "hardness_mode": null,
      "hardness_strength": 0.0
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 32
- `num_train_epochs`: 5.0
- `learning_rate`: 1e-05
- `warmup_steps`: 0.05
- `weight_decay`: 0.01
- `fp16`: True
- `per_device_eval_batch_size`: 32
- `data_seed`: 42
- `dataloader_num_workers`: 2
- `batch_sampler`: no_duplicates

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 32
- `num_train_epochs`: 5.0
- `max_steps`: -1
- `learning_rate`: 1e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0.05
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.01
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1.0
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: True
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: None
- `trackio_bucket_id`: None
- `trackio_static_space_id`: None
- `per_device_eval_batch_size`: 32
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: 42
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 2
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_static_graph`: None
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: None
- `fsdp_config`: None
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `warmup_ratio`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: no_duplicates
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch | Step | Training Loss | Validation Loss |
|:-----:|:----:|:-------------:|:---------------:|
| 1.0   | 105  | 0.0536        | 0.0201          |
| 2.0   | 210  | 0.0129        | 0.0089          |
| 3.0   | 315  | 0.0077        | 0.0049          |
| 4.0   | 420  | 0.0073        | 0.0037          |
| 5.0   | 525  | 0.0049        | 0.0034          |


### Training Time
- **Training**: 1.3 minutes

### Framework Versions
- Python: 3.12.13
- Sentence Transformers: 5.5.1
- Transformers: 5.12.0
- PyTorch: 2.11.0+cu128
- Accelerate: 1.14.0
- Datasets: 4.0.0
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{oord2019representationlearningcontrastivepredictive,
      title={Representation Learning with Contrastive Predictive Coding},
      author={Aaron van den Oord and Yazhe Li and Oriol Vinyals},
      year={2019},
      eprint={1807.03748},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1807.03748},
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->