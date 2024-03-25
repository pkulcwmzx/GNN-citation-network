入口在main.py。不同的实验设置也通过main开头的那些参数进行修改。      

目前很通俗易懂。      

主要可以看看datasets的文件。      

# TODO      
1. 不同噪声的添加方法。      
2. 不同难度的acc的统计。      
3. 折线图的绘制。      
4. llm进行评估的prompt和代码。      
5. 不同的prompt的方法的prompt。      
6. 文本类任务的难度控制、噪声控制。（TODO）      



# `datasets` Directory  Structure
## `math`
### `cmath_attn.json`

- based on `cmath_position`
- `Llama2-13B-chat-hf`
- List of Dicts. Each represents a sample.
    - `attn_score`: 
        - `$layer_num`: 
            - `out2inst_sum`: 
            An array of `[output_len]` (Output including additional texts `Answer: $golden`). Each element is 
            $$\sum_{\text{instruction token } i} Attn(\text{cur output token}\rarr i)$$
            - `out2inst_mean`:
            An array of `[output_len]` (Output including additional texts `Answer: $golden`). Each element is 
            $$\frac{\sum_{\text{instruction token } i} Attn(\text{cur output token}\rarr i)}{\text{instruction token}}$$
            - `out2ques_sum` & `out2ques_mean`
            - `out2dist_sum` & `out2dist_mean`
            - `instVSdist_sum`: 
            An array of `[output_len]` (Output including additional texts `Answer: $golden`). Each element is 
            $$\frac{\text{out2inst\sum}}{\text{out2dist\sum}}$$
            - `instVSdist_mean`:
            An array of `[output_len]` (Output including additional texts `Answer: $golden`). Each element is $$\frac{\sum_{\text{instruction token } i} Attn(\text{cur output token}\rarr i) / \text{instruction token}}{\sum_{\text{distraction token } j} Attn(\text{cur output token}\rarr j) / \text{distraction token}}$$
            - `quesVSdist_sum`
            - `quesVSdist_mean`
            > if there is no distraction, `NaN` is returned.
            
            > In fact, the attention score of each output token is not normalized over [inst，ques,dist] (because of other output tokens before cur output token). So the `{$typeA}VS{$typeB}` may not make sense.
    - `sailency_score`: same structure as `attn_score`
