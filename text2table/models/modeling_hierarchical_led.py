# Most of the code is taken from Huggingface LED implementation: 
# https://github.com/huggingface/transformers/blob/v4.20.1/src/transformers/models/led/modeling_led.py
# _prepare_encoder_decoder_kwargs_for_generation function is taken from transformers.generation_utils.py: 
# https://github.com/huggingface/transformers/blob/v4.21.1/src/transformers/generation_utils.py#L379
# In compliance with the Apache License, Version 2.0, I have modified the code to fit the needs of this project.
#
# Copyright 2021 Iz Beltagy, Matthew E. Peters, Arman Cohan and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pytorch hierarchical LED model for conditional generation"""

from transformers import LEDForConditionalGeneration
from transformers.models.led.modeling_led import (
    shift_tokens_right, 
    LEDSeq2SeqLMOutput, 
    LEDSeq2SeqModelOutput,
    LEDEncoderBaseModelOutput,
)
from transformers.utils import logging
import torch
from torch.nn import CrossEntropyLoss, AvgPool1d
from torch.nn.utils.rnn import pad_sequence


# The parts of the code that are not modified are not commented. Please refer to the original code for more information.
logger = logging.get_logger(__name__)

# Since our project only needs seq2seq model, this class inherit from the LEDForConditionalGeneration class.
class HierarchicalLEDForConditionalGeneration(LEDForConditionalGeneration):

    # avgpool_size is the size of the average pooling layer.
    def __init__(self, config, avgpool_size = 10):
        super().__init__(config)
        self.avgpool = AvgPool1d(avgpool_size)
    
    # The purpose of override this function is that we have to replicate 
    # the custom behavior about the encoder in the forward function in order to
    # make the evaluation flow function properly.
    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor, model_kwargs, model_input_name = None
    ):
        # 1. get encoder
        encoder = self.get_encoder()

        # 2. prepare encoder args and encoder kwargs from model kwargs
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor

        # assign a pointer to each encoder kwargs in order to make the following code stay the same as
        # the corresponding part in the forward function.
        input_ids = encoder_kwargs[model_input_name]
        attention_mask = encoder_kwargs["attention_mask"]
        global_attention_mask = encoder_kwargs["global_attention_mask"]
        head_mask = encoder_kwargs["head_mask"] if "head_mask" in encoder_kwargs else None
        inputs_embeds = encoder_kwargs["inputs_embeds"] if "inputs_embeds" in encoder_kwargs else None
        output_attentions = encoder_kwargs["output_attentions"]
        output_hidden_states = encoder_kwargs["output_hidden_states"]
        return_dict = encoder_kwargs["return_dict"]

        # The list contains all the encoder outputs of the clinical text of each category.
        encoder_outputs_list = []
        # get the encoder outputs of each group of text sequences in a batch.
        for i in range(len(input_ids)):
            # This is the index after which everything should be ignored.
            pad_index = len(input_ids[i])
            # if the attention mask is a tensor of all 0s, 
            # then the index is set to the index of the tensor.
            for j in range(len(input_ids[i])):
                if attention_mask[i][j].sum() == 0:
                    pad_index = j
                    break
            encoder_outputs_list.append(encoder(
                input_ids=input_ids[i][:pad_index],
                attention_mask=attention_mask[i][:pad_index],
                global_attention_mask=global_attention_mask[i][:pad_index],
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            ))
        
        # The list contains the pooled and concatenated encoder outputs. It is an unpadded batch.
        encoder_outputs_batch_list = []
        for i in range(len(encoder_outputs_list)):
            # split the encoder output batch into individual sequences.
            sequences = encoder_outputs_list[i][0].split(1)
            
            # get the attention mask of each individual sequence.
            sequences_mask = attention_mask[i].split(1)
            
            # get the pooled encoder outputs of each individual sequence and concatenate them.
            for j in range(len(sequences)):
                # get the pooled encoder outputs of each individual sequence. All the pad tokens are removed.
                avg = self.avgpool(sequences[j][:, :sequences_mask[j].sum(), :].permute(0, 2, 1)).permute(0, 2, 1)
                # concatenate the pooled encoder outputs of each individual sequence.
                # if it is the first part of the final sequence, then it is added to the batch.
                # Otherwise, it is concatenated to the sequence.
                if j == 0:
                    encoder_outputs_batch_list.append(avg)
                else:
                    encoder_outputs_batch_list[i] = torch.cat((encoder_outputs_batch_list[i], avg), dim=1)    
        
        # eliminiate the first dimension which is the batch dimension. The batch dimension is 1.
        for i in range(len(encoder_outputs_batch_list)):
            encoder_outputs_batch_list[i] = encoder_outputs_batch_list[i].squeeze(0)
        
        # pad the encoder output batch and convert it to a tensor.
        encoder_outputs_ = pad_sequence(encoder_outputs_batch_list, batch_first=True, padding_value=1).to(input_ids.device)
        
        # The list contains the attention masks of the encoder outputs.
        attention_mask_batch_list = []
        # get the attention masks of the encoder outputs 
        # by using ones to set the attention mask of the non-pad tokens to 1.
        for i in range(len(encoder_outputs_batch_list)):
            attention_mask_batch_list.append(torch.ones(len(encoder_outputs_batch_list[i])).to(attention_mask.device))
        
        # pad the attention masks and convert it to a tensor. 
        # By padding them, we set the attention mask of the pad tokens to 0.
        attention_mask_ = pad_sequence(attention_mask_batch_list, batch_first=True).to(attention_mask.device)

        model_kwargs["encoder_outputs"] = LEDEncoderBaseModelOutput(last_hidden_state=encoder_outputs_)
        model_kwargs["attention_mask"] = attention_mask_

        return model_kwargs
    
    # input_ids should be a list of batchs of token ids. Each batch corresponds to one kind of clinical text.
    # The length of the list should be equal to the number of categories of the clinical text.
    # attention_mask should be a list of batchs of attention mask. Each batch corresponds to one kind of clinical text.
    # The length of the list should be equal to the number of categories of the clinical text.
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        global_attention_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        # The following code is the core of the hierarchical LED model.
        if encoder_outputs is None:
            # The list contains all the encoder outputs of the clinical text of each category.
            encoder_outputs_list = []
            # get the encoder outputs of each group of text sequences in a batch.
            for i in range(len(input_ids)):
                # This is the index after which everything should be ignored.
                pad_index = len(input_ids[i])
                # if the attention mask is a tensor of all 0s, 
                # then the index is set to the index of the tensor.
                for j in range(len(input_ids[i])):
                    if attention_mask[i][j].sum() == 0:
                        pad_index = j
                        break

                encoder_outputs_list.append(self.led.encoder(
                    input_ids=input_ids[i][:pad_index],
                    attention_mask=attention_mask[i][:pad_index],
                    global_attention_mask=global_attention_mask[i][:pad_index],
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                ))
            
            # The list contains the pooled and concatenated encoder outputs. It is an unpadded batch.
            encoder_outputs_batch_list = []
            for i in range(len(encoder_outputs_list)):
                # split the encoder output batch into individual sequences.
                sequences = encoder_outputs_list[i][0].split(1)
                
                # get the attention mask of each individual sequence.
                sequences_mask = attention_mask[i].split(1)
                
                # get the pooled encoder outputs of each individual sequence and concatenate them.
                for j in range(len(sequences)):
                    # get the pooled encoder outputs of each individual sequence. All the pad tokens are removed.
                    avg = self.avgpool(sequences[j][:, :sequences_mask[j].sum(), :].permute(0, 2, 1)).permute(0, 2, 1)
                    # concatenate the pooled encoder outputs of each individual sequence.
                    # if it is the first part of the final sequence, then it is added to the batch.
                    # Otherwise, it is concatenated to the sequence.
                    if j == 0:
                        encoder_outputs_batch_list.append(avg)
                    else:
                        encoder_outputs_batch_list[i] = torch.cat((encoder_outputs_batch_list[i], avg), dim=1)    
            
            # eliminiate the first dimension which is the batch dimension. The batch dimension is 1.
            for i in range(len(encoder_outputs_batch_list)):
                encoder_outputs_batch_list[i] = encoder_outputs_batch_list[i].squeeze(0)
            
            # pad the encoder output batch and convert it to a tensor.
            encoder_outputs_ = pad_sequence(encoder_outputs_batch_list, batch_first=True, padding_value=1).to(input_ids.device)
            
            # The list contains the attention masks of the encoder outputs.
            attention_mask_batch_list = []
            # get the attention masks of the encoder outputs 
            # by using ones to set the attention mask of the non-pad tokens to 1.
            for i in range(len(encoder_outputs_batch_list)):
                attention_mask_batch_list.append(torch.ones(len(encoder_outputs_batch_list[i])).to(attention_mask.device))
            
            # pad the attention masks and convert it to a tensor. 
            # By padding them, we set the attention mask of the pad tokens to 0.
            attention_mask_ = pad_sequence(attention_mask_batch_list, batch_first=True).to(attention_mask.device)
        
        decoder_outputs = self.led.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            # Original code: encoder_hidden_states=encoder_outputs[0]
            encoder_hidden_states=encoder_outputs_ if "encoder_outputs_" in locals() else encoder_outputs[0],
            # Original code: encoder_attention_mask=attention_mask
            encoder_attention_mask=attention_mask_ if "attention_mask_" in locals() else attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            outputs = decoder_outputs + encoder_outputs
        
        outputs = LEDSeq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
        )
        
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return LEDSeq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
        )
        