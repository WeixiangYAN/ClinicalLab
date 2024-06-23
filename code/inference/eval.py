import re
import json
import time
import torch
import logging
import argparse
import platform

from tqdm import tqdm
from pathlib import Path

from evaluators.pulse import PULSEEvaluator
from evaluators.spark3 import Spark3Evaluator
from evaluators.yichat import YiChatEvaluator
from evaluators.claude3 import Claude3Evaluator
from evaluators.wingpt2 import WiNGPT2Evaluator
from evaluators.chatgpt import ChatGPTEvaluator
from evaluators.taiyillm import TaiyiLLMvaluator
from evaluators.chatglm3 import ChatGLM3Evaluator
from evaluators.qwenchat import QwenChatEvaluator
from evaluators.bianque2 import BianQue2Evaluator
from evaluators.geminipro import GeminiProEvaluator
from evaluators.bluelmchat import BlueLMChatEvaluator
from evaluators.discmedllm import DISCMedLLMEvaluator
from evaluators.huatuogpt2 import HuatuoGPT2Evaluator
from evaluators.internlm2chat import InternLM2ChatEvaluator
from evaluators.baichuan2chat import Baichuan2ChatEvaluator


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default=None,
                        choices=[
                            'gpt4',
                            'gpt3.5',
                            'geminipro',
                            'palm2',
                            'qwenchat',
                            'yichat',
                            'chatglm3',
                            'baichuan2chat',
                            'internlm2chat',
                            'discmedllm',
                            'bianque2',
                            'pulse',
                            'huatuogpt2',
                            'taiyillm',
                            'wingpt2',
                            'spark3',
                            'bluelmchat',
                            'claude3'
                        ], type=str)
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--cache_dir', default=None, type=str)
    parser.add_argument('--api_key', default=None, type=str)
    parser.add_argument('--api_secret', default=None, type=str)  # only for spark3
    parser.add_argument('--app_id', default=None, type=str)  # only for spark3
    parser.add_argument('--data_load_name', default=None, type=str)
    parser.add_argument('--result_save_name', default=None, type=str)
    parser.add_argument('--log_save_name', default=None, type=str)
    args = parser.parse_args()

    return args


def generate_hospital_guide(example):
    clinical_case_uid = example['clinical_case_uid']
    language = example['language']
    clinical_case_summary = example['clinical_case_summary']

    if language == 'zh':
        patient_information_pattern = r'患者基本信息：([\s\S]*?)主诉：'
        chief_complaint_pattern = r'主诉：([\s\S]*?)病史：'
    else:
        patient_information_pattern = r'Patient Basic Information:([\s\S]*?)Chief Complaint:'
        chief_complaint_pattern = r'Chief Complaint:([\s\S]*?)Medical History:'

    patient_information_matches = re.search(patient_information_pattern, clinical_case_summary, re.DOTALL)
    if patient_information_matches:
        patient_information_content = patient_information_matches.group(1).strip()
    else:
        logging.error('### [Failed to extract patient information]')
        raise Exception('Failed to extract patient information')

    chief_complaint_matches = re.search(chief_complaint_pattern, clinical_case_summary, re.DOTALL)
    if chief_complaint_matches:
        chief_complaint_content = chief_complaint_matches.group(1).strip()
    else:
        logging.error('### [Failed to extract chief complaint]')
        raise Exception('Failed to extract chief complain')

    logging.info('===> [hospital guide inference]...')
    logging.info('### [clinical case uid]: ' + str(clinical_case_uid))

    if language == 'zh':
        prompt = f'''您是一位经验丰富的医院导诊员，根据患者提供的性别、年龄和症状等信息，请从医院科室列表中选择一个患者最需要前往治疗的科室。
请确保您的回答只包含选中的一个科室，不要包含多余的内容。

以下是患者提供的信息：
{patient_information_content + chief_complaint_content}

以下是医院科室列表：
{clinical_department_zh_list}
您选中的科室：'''
    else:
        prompt = f'''You are an experienced hospital guide. Based on the information provided by the patient, including gender, age, and symptoms, please select a department from the list of hospital departments that the patient needs to visit for treatment most.
Please ensure that your response includes only one selected department without any additional content.

The following is the patient's information:
{patient_information_content + chief_complaint_content}

The following is the list of hospital departments:
{clinical_department_en_list}
Your selected department:'''
    prompt_tokens = evaluator.count_tokens(prompt)

    logging.info('### [prompt]: ' + str(prompt))
    logging.info('### [prompt tokens]: ' + str(prompt_tokens))

    try:
        response = evaluator.generate_text(prompt)

        logging.info('### [response]: ' + str(response))

        if response is None:
            response = ''
            response_tokens = 0
        else:
            response_tokens = evaluator.count_tokens(response)
        total_tokens = prompt_tokens + response_tokens

        logging.info('### [response tokens]: ' + str(response_tokens))
        logging.info('### [total tokens]: ' + str(total_tokens))

    except Exception as e:
        response = ''

        logging.error('### [Failed to generate text]: ' + e.__str__())

    example['prompt_hospital_guide'] = str(evaluator.format_prompt(prompt))
    example['predicted_clinical_department'] = response

    return example


def generate_clinical_diagnosis(example):
    clinical_case_uid = example['clinical_case_uid']
    language = example['language']
    clinical_department = example['clinical_department']
    clinical_case_summary = example['clinical_case_summary']

    logging.info('===> [clinical diagnosis inference]...')
    logging.info('### [clinical case uid]: ' + str(clinical_case_uid))

    if language == 'zh':
        prompt = f'''您是一位经验丰富的{clinical_department}临床医生，根据给定的病例摘要，请分析并提供一个专业、详细、全面的临床诊断，包含以下6个部分：
1. 初步诊断：分条列出患者可能存在的一种或多种疾病的名称。
2. 诊断依据：分条列出您做出初步诊断的依据。
3. 鉴别诊断：分条列出几种可能导致患者目前症状的疾病，并简要说明为何将它们排除。如果您认为不需要进行鉴别诊断，请直接回答“诊断明确，无需鉴别。”。
4. 主要诊断：一种对患者身体健康危害最大、最需要治疗的疾病的名称。
5. 治疗原则：概述治疗您给出的主要诊断的一般原则。
6. 治疗计划：分条列出初步的治疗计划。

请确保您的回答符合以下格式：
“初步诊断：<您提供的初步诊断>
诊断依据：<您提供的诊断依据>
鉴别诊断：<您提供的鉴别诊断>
主要诊断：<您提供的主要诊断>
治疗原则：<您提供的治疗原则>
治疗计划：<您提供的治疗计划>”

以下是给定的病例摘要：

{clinical_case_summary}

您提供的临床诊断：'''
        preliminary_diagnosis_pattern = r'初步诊断：([\s\S]*?)诊断依据：'
        diagnostic_basis_pattern = r'诊断依据：([\s\S]*?)鉴别诊断：'
        differential_diagnosis_pattern = r'鉴别诊断：([\s\S]*?)主要诊断：'
        principal_diagnosis_pattern = r'主要诊断：([\s\S]*?)治疗原则：'
        therapeutic_principle_pattern = r'治疗原则：([\s\S]*?)治疗计划：'
        treatment_plan_pattern = r'治疗计划：([\s\S]*?)$'
    else:
        prompt = f'''You are an experienced {clinical_department} clinician. Based on the given clinical case summary, please analyze and provide a professional, detailed, and comprehensive clinical diagnosis, including the following 6 parts:
1. Preliminary Diagnosis: List the names of one or multiple diseases that the patient might have.
2. Diagnostic Basis: List the basis for your preliminary diagnosis.
3. Differential Diagnosis: List several diseases that could cause the patient's current symptoms and briefly explain why you exclude them. If you believe differential diagnosis is unnecessary, please directly response "The diagnosis is clear and no differentiation is needed."
4. Principal Diagnosis: The name of a disease that is most harmful to the patient's physical health and needs immediate treatment.
5. Therapeutic Principle: Outline the general principles for treating your principal diagnosis.
6. Treatment Plan: List the rough treatment plan.

Please ensure that your response follows this format:
"Preliminary Diagnosis: <Your Preliminary Diagnosis>
Diagnostic Basis: <Your Diagnostic Basis>
Differential Diagnosis: <Your Differential Diagnosis>
Principal Diagnosis: <Your Principal Diagnosis>
Therapeutic Principle: <Your Therapeutic Principle>
Treatment Plan: <Your Treatment Plan>"

The following is the given clinical case summary:
{clinical_case_summary}

Your clinical diagnosis:'''
        preliminary_diagnosis_pattern = r'Preliminary Diagnosis:([\s\S]*?)Diagnostic Basis:'
        diagnostic_basis_pattern = r'Diagnostic Basis:([\s\S]*?)Differential Diagnosis:'
        differential_diagnosis_pattern = r'Differential Diagnosis:([\s\S]*?)Principal Diagnosis:'
        principal_diagnosis_pattern = r'Principal Diagnosis:([\s\S]*?)Therapeutic Principle:'
        therapeutic_principle_pattern = r'Therapeutic Principle:([\s\S]*?)Treatment Plan:'
        treatment_plan_pattern = r'Treatment Plan:([\s\S]*?)$'

    prompt_tokens = evaluator.count_tokens(prompt)

    logging.info('### [prompt]: ' + str(prompt))
    logging.info('### [prompt tokens]: ' + str(prompt_tokens))

    try:
        response = evaluator.generate_text(prompt)

        logging.info('### [response]: ' + str(response))

        if response is None:
            response = ''
            response_tokens = 0
        else:
            response_tokens = evaluator.count_tokens(response)
        total_tokens = prompt_tokens + response_tokens

        logging.info('### [response tokens]: ' + str(response_tokens))
        logging.info('### [total tokens]: ' + str(total_tokens))

    except Exception as e:
        response = ''

        logging.error('### [Failed to generate text]: ' + e.__str__())

    example['prompt_clinical_diagnosis'] = str(evaluator.format_prompt(prompt))
    example['predicted_clinical_diagnosis'] = response

    predicted_preliminary_diagnosis = ''
    predicted_diagnostic_basis = ''
    predicted_differential_diagnosis = ''
    predicted_principal_diagnosis = ''
    predicted_therapeutic_principle = ''
    predicted_treatment_plan = ''

    preliminary_diagnosis_match = re.search(preliminary_diagnosis_pattern, response, re.DOTALL)
    if preliminary_diagnosis_match:
        predicted_preliminary_diagnosis = preliminary_diagnosis_match.group(1).strip()
        logging.info('### [preliminary diagnosis]: ' + str(predicted_preliminary_diagnosis))
    else:
        logging.error(f'### [Failed to match preliminary diagnosis]')

    diagnostic_basis_match = re.search(diagnostic_basis_pattern, response, re.DOTALL)
    if diagnostic_basis_match:
        predicted_diagnostic_basis = diagnostic_basis_match.group(1).strip()
        logging.info('### [diagnostic basis]: ' + str(predicted_diagnostic_basis))
    else:
        logging.error(f'### [Failed to match diagnostic basis]')

    differential_diagnosis_match = re.search(differential_diagnosis_pattern, response, re.DOTALL)
    if differential_diagnosis_match:
        predicted_differential_diagnosis = differential_diagnosis_match.group(1).strip()
        logging.info('### [differential diagnosis]: ' + str(predicted_differential_diagnosis))
    else:
        logging.error(f'### [Failed to match differential diagnosis]')

    principal_diagnosis_match = re.search(principal_diagnosis_pattern, response, re.DOTALL)
    if principal_diagnosis_match:
        predicted_principal_diagnosis = principal_diagnosis_match.group(1).strip()
        logging.info('### [principal diagnosis]: ' + str(predicted_principal_diagnosis))
    else:
        logging.error(f'### [Failed to match principal diagnosis]')

    therapeutic_principle_match = re.search(therapeutic_principle_pattern, response, re.DOTALL)
    if therapeutic_principle_match:
        predicted_therapeutic_principle = therapeutic_principle_match.group(1).strip()
        logging.info('### [therapeutic principle]: ' + str(predicted_therapeutic_principle))
    else:
        logging.error(f'### [Failed to match therapeutic principle]')

    treatment_plan_match = re.search(treatment_plan_pattern, response, re.DOTALL)
    if treatment_plan_match:
        predicted_treatment_plan = treatment_plan_match.group(1).strip()
        logging.info('### [treatment plan]: ' + str(predicted_treatment_plan))
    else:
        logging.error(f'### [Failed to match treatment plan]')

    example['predicted_preliminary_diagnosis'] = predicted_preliminary_diagnosis
    example['predicted_diagnostic_basis'] = predicted_diagnostic_basis
    example['predicted_differential_diagnosis'] = predicted_differential_diagnosis
    example['predicted_principal_diagnosis'] = predicted_principal_diagnosis
    example['predicted_therapeutic_principle'] = predicted_therapeutic_principle
    example['predicted_treatment_plan'] = predicted_treatment_plan

    return example


def generate_imaging_diagnosis(example):
    clinical_case_uid = example['clinical_case_uid']
    language = example['language']
    imageological_examination = example['imageological_examination']

    logging.info('===> [imaging diagnosis inference]...')
    logging.info('### [clinical case uid]: ' + str(clinical_case_uid))

    if isinstance(imageological_examination, dict):
        for imageological_examination_part_feature in imageological_examination.keys():
            findings = imageological_examination[imageological_examination_part_feature]['findings']
            if language == 'zh':
                imageological_examination_part_name = imageological_examination_part_feature_to_name_zh_dict[imageological_examination_part_feature]
                prompt = f'''您是一位经验丰富的放射科医生，根据给定的{imageological_examination_part_name}检查报告中的影像所见部分，请分析并分条列出专业的影像诊断。
请确保您的回答高度简洁。

以下是给定的影像所见：
{findings}

您提供的影像诊断：'''
            else:
                imageological_examination_part_name = imageological_examination_part_feature_to_name_en_dict[imageological_examination_part_feature]
                prompt = f'''You are an experienced radiologist. Based on the findings section of the given {imageological_examination_part_name} examination report, please analyze and list professional impression.
Please ensure that your response is highly concise.

The following is the given findings:
{findings}

Your impression:'''
            prompt_tokens = evaluator.count_tokens(prompt)

            logging.info('### [prompt]: ' + str(prompt))
            logging.info('### [prompt tokens]: ' + str(prompt_tokens))

            try:
                response = evaluator.generate_text(prompt)

                logging.info('### [response]: ' + str(response))

                if response is None:
                    response = ''
                    response_tokens = 0
                else:
                    response_tokens = evaluator.count_tokens(response)
                total_tokens = prompt_tokens + response_tokens

                logging.info('### [response tokens]: ' + str(response_tokens))
                logging.info('### [total tokens]: ' + str(total_tokens))

            except Exception as e:
                response = ''

                logging.error('### [Failed to generate text]: ' + e.__str__())

            imageological_examination[imageological_examination_part_feature]['prompt_impression'] = str(evaluator.format_prompt(prompt))
            imageological_examination[imageological_examination_part_feature]['predicted_impression'] = response

    return example


def main():
    with open(data_load_path, mode='r', encoding='utf-8') as file:
        dataset = json.load(file)

    for index in tqdm(range(len(dataset))):
        dataset[index] = generate_hospital_guide(dataset[index])
        dataset[index] = generate_clinical_diagnosis(dataset[index])
        dataset[index] = generate_imaging_diagnosis(dataset[index])

        with open(result_save_path, mode='w', encoding='utf-8') as file:
            json.dump(dataset, file, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    args = parse_arguments()

    timestamp = time.strftime('%Y-%m-%d-%H-%M-%S')
    data_dir = Path(__file__).parent.parent / Path('data')
    if not data_dir.is_dir():
        data_dir.mkdir(parents=True, exist_ok=True)
    data_load_path = data_dir / Path(args.data_load_name)
    result_dir = Path(__file__).parent.parent / Path('results')
    if not result_dir.is_dir():
        result_dir.mkdir(parents=True, exist_ok=True)
    if args.result_save_name is None:
        result_save_path = result_dir / Path(args.data_load_name.split('.')[0].replace('data', 'result') + f'_{args.model_name}({timestamp}).json')
    else:
        result_save_path = result_dir / Path(args.result_save_name.split('.')[0] + f'({timestamp}).json')
    log_dir = Path(__file__).parent.parent / Path('logs')
    if not log_dir.is_dir():
        log_dir.mkdir(parents=True, exist_ok=True)
    if args.log_save_name is None:
        log_save_path = log_dir / Path(args.data_load_name.split('.')[0].replace('data', 'result') + f'_{args.model_name}({timestamp}).log')
    else:
        log_save_path = log_dir / Path(args.log_save_name.split('.')[0] + f'({timestamp}).log')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s - %(filename)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(filename=log_save_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else platform.processor()

    if args.model_name == 'gpt4':
        if args.model_path is None:
            pretrained_model_name_or_path = 'gpt-4-1106-preview'
        else:
            pretrained_model_name_or_path = args.model_path
        evaluator = ChatGPTEvaluator(pretrained_model_name_or_path=pretrained_model_name_or_path, api_key=args.api_key)
    elif args.model_name == 'gpt3.5':
        if args.model_path is None:
            pretrained_model_name_or_path = 'gpt-3.5-turbo-1106'
        else:
            pretrained_model_name_or_path = args.model_path
        evaluator = ChatGPTEvaluator(pretrained_model_name_or_path=pretrained_model_name_or_path, api_key=args.api_key)
    elif args.model_name == 'geminipro':
        if args.model_path is None:
            pretrained_model_name_or_path = 'models/gemini-pro'
        else:
            pretrained_model_name_or_path = args.model_path
        evaluator = GeminiProEvaluator(pretrained_model_name_or_path=pretrained_model_name_or_path, api_key=args.api_key)
    elif args.model_name == 'claude3':
        if args.model_path is None:
            pretrained_model_name_or_path = 'claude-3-haiku-20240307'
        else:
            pretrained_model_name_or_path = args.model_path
        evaluator = Claude3Evaluator(pretrained_model_name_or_path=pretrained_model_name_or_path, api_key=args.api_key)
    elif args.model_name == 'spark3':
        if args.model_path is None:
            pretrained_model_name_or_path = 'v3.0'
        else:
            pretrained_model_name_or_path = args.model_path
        evaluator = Spark3Evaluator(pretrained_model_name_or_path=pretrained_model_name_or_path, app_id=args.app_id, api_secret=args.api_secret, api_key=args.api_key)
    elif args.model_name == 'qwenchat':
        if args.model_path is None:
            pretrained_model_name_or_path = 'qwen-72b-chat'
        else:
            pretrained_model_name_or_path = args.model_path
        evaluator = QwenChatEvaluator(pretrained_model_name_or_path=pretrained_model_name_or_path, api_key=args.api_key)
    elif args.model_name == 'yichat':
        if args.model_path is None:
            pretrained_model_name_or_path = ''
        else:
            pretrained_model_name_or_path = args.model_path
        evaluator = YiChatEvaluator(pretrained_model_name_or_path=pretrained_model_name_or_path, cache_dir=args.cache_dir)
    elif args.model_name == 'chatglm3':
        if args.model_path is None:
            pretrained_model_name_or_path = ''
        else:
            pretrained_model_name_or_path = args.model_path
        evaluator = ChatGLM3Evaluator(pretrained_model_name_or_path=pretrained_model_name_or_path, cache_dir=args.cache_dir)
    elif args.model_name == 'baichuan2chat':
        if args.model_path is None:
            pretrained_model_name_or_path = ''
        else:
            pretrained_model_name_or_path = args.model_path
        evaluator = Baichuan2ChatEvaluator(pretrained_model_name_or_path=pretrained_model_name_or_path, cache_dir=args.cache_dir)
    elif args.model_name == 'internlm2chat':
        if args.model_path is None:
            pretrained_model_name_or_path = ''
        else:
            pretrained_model_name_or_path = args.model_path
        evaluator = InternLM2ChatEvaluator(pretrained_model_name_or_path=pretrained_model_name_or_path, cache_dir=args.cache_dir)
    elif args.model_name == 'discmedllm':
        if args.model_path is None:
            pretrained_model_name_or_path = ''
        else:
            pretrained_model_name_or_path = args.model_path
        evaluator = DISCMedLLMEvaluator(pretrained_model_name_or_path=pretrained_model_name_or_path, cache_dir=args.cache_dir)
    elif args.model_name == 'bianque2':
        if args.model_path is None:
            pretrained_model_name_or_path = ''
        else:
            pretrained_model_name_or_path = args.model_path
        evaluator = BianQue2Evaluator(pretrained_model_name_or_path=pretrained_model_name_or_path, cache_dir=args.cache_dir)
    elif args.model_name == 'pulse':
        if args.model_path is None:
            pretrained_model_name_or_path = ''
        else:
            pretrained_model_name_or_path = args.model_path
        evaluator = PULSEEvaluator(pretrained_model_name_or_path=pretrained_model_name_or_path, cache_dir=args.cache_dir)
    elif args.model_name == 'huatuogpt2':
        if args.model_path is None:
            pretrained_model_name_or_path = ''
        else:
            pretrained_model_name_or_path = args.model_path
        evaluator = HuatuoGPT2Evaluator(pretrained_model_name_or_path=pretrained_model_name_or_path, cache_dir=args.cache_dir)
    elif args.model_name == 'taiyillm':
        if args.model_path is None:
            pretrained_model_name_or_path = ''
        else:
            pretrained_model_name_or_path = args.model_path
        evaluator = TaiyiLLMvaluator(pretrained_model_name_or_path=pretrained_model_name_or_path, cache_dir=args.cache_dir)
    elif args.model_name == 'wingpt2':
        if args.model_path is None:
            pretrained_model_name_or_path = ''
        else:
            pretrained_model_name_or_path = args.model_path
        evaluator = WiNGPT2Evaluator(pretrained_model_name_or_path=pretrained_model_name_or_path, cache_dir=args.cache_dir)
    elif args.model_name == 'bluelmchat':
        if args.model_path is None:
            pretrained_model_name_or_path = ''
        else:
            pretrained_model_name_or_path = args.model_path
        evaluator = BlueLMChatEvaluator(pretrained_model_name_or_path=pretrained_model_name_or_path, cache_dir=args.cache_dir)
    else:
        pretrained_model_name_or_path = None
        evaluator = None
        logger.error('### [model name not found]')
        exit()

    logging.info('### [device]: ' + str(device))
    logging.info('### [device name]: ' + str(device_name))
    logging.info('### [model name]: ' + str(args.model_name))
    logging.info('### [model path]: ' + str(pretrained_model_name_or_path))

    imageological_examination_part_name_zh_to_feature_dict = {
        'CT平扫': 'plain_computed_tomography_scan',
        'CT增强': 'contrast_computed_tomography_scan',
        'CT平扫+增强': 'plain_computed_tomography_scan+contrast_computed_tomography_scan',
        '彩色多普勒超声': 'color_doppler_ultrasound',
        '乳腺钼靶': 'mammography',
        '磁共振平扫': 'plain_magnetic_resonance_imaging_scan',
        '磁共振增强': 'contrast_magnetic_resonance_imaging_scan',
        '磁共振平扫+增强': 'plain_magnetic_resonance_imaging_scan+contrast_magnetic_resonance_imaging_scan',
        '磁共振血管造影': 'magnetic_resonance_angiography',
        '磁共振平扫+血管造影': 'plain_magnetic_resonance_imaging_scan+magnetic_resonance_angiography',
        '磁共振水成像': 'magnetic_resonance_hydrography',
        '磁共振平扫+水成像': 'plain_magnetic_resonance_imaging_scan+magnetic_resonance_hydrography',
        '数字X线摄影': 'digital_radiography',
        'CT血管造影': 'computed_tomography_angiography',
        '超声心动图': 'echocardiogram',
        '冠状动脉造影': 'coronary_arteriography',
        '食道、胃、十二指肠镜': 'esophagogastroduodenoscopy',
        '甲状旁腺显像': 'parathyroid_imaging',
        '电子鼻咽喉镜': 'nasopharyngoscope',
        '消化道造影': 'gastrointestinal_tract_radiography'
    }
    imageological_examination_part_feature_to_name_zh_dict = {value: key for key, value in imageological_examination_part_name_zh_to_feature_dict.items()}

    imageological_examination_part_feature_to_name_en_dict = {
        'plain CT scan': 'plain_computed_tomography_scan',
        'contrast CT scan': 'contrast_computed_tomography_scan',
        'plain CT scan and contrast CT scan': 'plain_computed_tomography_scan+contrast_computed_tomography_scan',
        'color doppler ultrasound': 'color_doppler_ultrasound',
        'mammography': 'mammography',
        'plain MRI scan': 'plain_magnetic_resonance_imaging_scan',
        'contrast MRI scan': 'contrast_magnetic_resonance_imaging_scan',
        'plain MRI scan and contrast MRI scan': 'plain_magnetic_resonance_imaging_scan+contrast_magnetic_resonance_imaging_scan',
        'MRA': 'magnetic_resonance_angiography',
        'plain MRI scan and MRA': 'plain_magnetic_resonance_imaging_scan+magnetic_resonance_angiography',
        'MR hydrography': 'magnetic_resonance_hydrography',
        'plain MRI scan and MR hydrography': 'plain_magnetic_resonance_imaging_scan+magnetic_resonance_hydrography',
        'digital radiography': 'digital_radiography',
        'CT angiography': 'computed_tomography_angiography',
        'echocardiogram': 'echocardiogram',
        'coronary arteriography': 'coronary_arteriography',
        'esophagogastroduodenoscopy': 'esophagogastroduodenoscopy',
        'parathyroid imaging': 'parathyroid_imaging',
        'nasopharyngoscope': 'nasopharyngoscope',
        'gastrointestinal tract radiography': 'gastrointestinal_tract_radiography'
    }
    imageological_examination_part_feature_to_name_en_dict = {value: key for key, value in imageological_examination_part_feature_to_name_en_dict.items()}

    clinical_department_zh_to_en_dict = {
        '乳腺外科': 'breast surgical department',
        '产科': 'obstetrics department',
        '儿科': 'pediatrics department',
        '内分泌内科': 'endocrinology department',
        '呼吸内科': 'respiratory medicine department',
        '妇科': 'gynecology department',
        '心脏外科': 'cardiac surgical department',
        '心血管内科': 'cardiovascular medicine department',
        '泌尿外科': 'urinary surgical department',
        '消化内科': 'gastroenterology department',
        '甲状腺外科': 'thyroid surgical department',
        '疝外科': 'hernia surgical department',
        '神经内科': 'neurology department',
        '神经外科': 'neurosurgery department',
        '耳鼻咽喉头颈外科': 'otolaryngology head and neck surgical department',
        '肛门结直肠外科': 'anus and intestine surgical department',
        '肝胆胰外科': 'hepatobiliary and pancreas surgical department',
        '肾内科': 'nephrology department',
        '胃肠外科': 'gastrointestinal surgical department',
        '胸外科': 'thoracic surgical department',
        '血液内科': 'hematology department',
        '血管外科': 'vascular surgical department',
        '骨科': 'orthopedics department',
    }
    clinical_department_zh_list = ''
    for _ in clinical_department_zh_to_en_dict.keys():
        clinical_department_zh_list += f'- {_}\n'
    clinical_department_en_list = ''
    for _ in clinical_department_zh_to_en_dict.values():
        clinical_department_en_list += f'- {_}\n'

    main()
