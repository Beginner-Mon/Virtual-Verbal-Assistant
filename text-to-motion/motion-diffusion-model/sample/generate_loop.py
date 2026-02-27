# This code is based on https://github.com/openai/guided-diffusion
"""
Generate 6 motion samples, prompting for a new text input after each sample.
"""
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_saved_model
from utils import dist_util
from utils.sampler_util import ClassifierFreeSampleModel, AutoRegressiveSampler
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
import shutil
from data_loaders.tensors import collate
from moviepy.editor import clips_array


def main(args=None):
    if args is None:
        # args is None unless this method is called from another function (e.g. during training)
        args = generate_args()
    fixseed(args.seed)
    out_path = args.output_dir
    n_joints = 22 if args.dataset == 'humanml' else 21
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    max_frames = 196 if args.dataset in ['kit', 'humanml'] else 60
    fps = 12.5 if args.dataset == 'kit' else 20
    n_frames = min(max_frames, int(args.motion_length * fps))

    # Always text-to-motion in this looped version
    args.num_samples = 6
    args.batch_size = 6
    args.num_repetitions = 1

    dist_util.setup_dist(args.device)
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'samples_{}_{}_seed{}_loop'.format(name, niter, args.seed))

    print('Loading dataset...')
    data = load_dataset(args, max_frames, n_frames)

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    sample_fn = diffusion.p_sample_loop
    if args.autoregressive:
        sample_cls = AutoRegressiveSampler(args, sample_fn, n_frames)
        sample_fn = sample_cls.sample

    print(f"Loading checkpoints from [{args.model_path}]...")
    load_saved_model(model, args.model_path, use_avg=args.use_ema)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    motion_shape = (args.batch_size, model.njoints, model.nfeats, n_frames)

    prompt_count = 0
    while True:
        prompt = input("Enter prompt (empty to stop): ").strip()
        if prompt == '':
            print("Empty prompt. Stopping.")
            break
        prompt_count += 1

        collate_args = [{'inp': torch.zeros(n_frames), 'tokens': None, 'lengths': n_frames, 'text': prompt}] * args.num_samples
        _, model_kwargs = collate(collate_args)

        # Move conditioning to device
        model_kwargs['y'] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val
                             for key, val in model_kwargs['y'].items()}

        # add CFG scale to batch
        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

        if 'text' in model_kwargs['y'].keys():
            # encoding once instead of each iteration saves time
            model_kwargs['y']['text_embed'] = model.encode_text(model_kwargs['y']['text'])

        print(f'### Sampling [prompt #{prompt_count}]')
        sample = sample_fn(
            model,
            motion_shape,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )

        # Recover XYZ *positions* from HumanML3D vector representation
        if model.data_rep == 'hml_vec':
            n_joints = 22 if sample.shape[1] == 263 else 21
            sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
            sample = recover_from_ric(sample, n_joints)
            sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

        rot2xyz_pose_rep = 'xyz' if model.data_rep in ['xyz', 'hml_vec'] else model.data_rep
        rot2xyz_mask = None if rot2xyz_pose_rep == 'xyz' else model_kwargs['y']['mask'].reshape(args.batch_size, n_frames).bool()
        sample = model.rot2xyz(x=sample, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
                               jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None,
                               get_rotations_back=False)

        all_text = [prompt] * args.num_samples
        all_motions = [sample.cpu().numpy()]
        _len = model_kwargs['y']['lengths'].cpu().numpy()
        all_lengths = [_len]

        print(f"created {len(all_motions) * args.batch_size} samples")

        # save per-prompt outputs in its own folder
        safe_prompt = ''.join([c if c.isalnum() or c in ['_', '-', '.'] else '_' for c in prompt])[:80]
        prompt_out_path = os.path.join(out_path, f"prompt_{prompt_count:02d}_{safe_prompt}")
        if os.path.exists(prompt_out_path):
            shutil.rmtree(prompt_out_path)
        os.makedirs(prompt_out_path)

        all_motions = np.concatenate(all_motions, axis=0)
        all_text = all_text[:len(all_motions)]
        all_lengths = np.concatenate(all_lengths, axis=0)[:len(all_motions)]

        npy_path = os.path.join(prompt_out_path, 'results.npy')
        print(f"saving results file to [{npy_path}]")
        np.save(npy_path,
                {'motion': all_motions, 'text': all_text, 'lengths': all_lengths,
                 'num_samples': len(all_text), 'num_repetitions': 1})

        text_file_content = '\n'.join(all_text)
        with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
            fw.write(text_file_content)
        with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
            fw.write('\n'.join([str(l) for l in all_lengths]))

        print(f"saving visualizations to [{prompt_out_path}]...")
        skeleton = paramUtil.kit_kinematic_chain if args.dataset == 'kit' else paramUtil.t2m_kinematic_chain

        sample_print_template, row_print_template, all_print_template, \
        sample_file_template, row_file_template, all_file_template = construct_template_variables(args.unconstrained)
        max_vis_samples = 6
        num_vis_samples = min(len(all_text), max_vis_samples)
        animations = np.empty(shape=(num_vis_samples, 1), dtype=object)
        max_length = max(all_lengths)

        for sample_i in range(num_vis_samples):
            caption = all_text[sample_i]
            length = all_lengths[sample_i]
            motion = all_motions[sample_i].transpose(2, 0, 1)[:max_length]
            if motion.shape[0] > length:
                motion[length:-1] = motion[length - 1]

            save_file = sample_file_template.format(sample_i, 0)
            animation_save_path = os.path.join(prompt_out_path, save_file)
            animations[sample_i, 0] = plot_3d_motion(animation_save_path,
                                                     skeleton, motion, dataset=args.dataset, title=caption,
                                                     fps=fps, gt_frames=[])

        save_multiple_samples(prompt_out_path, {'all': all_file_template}, animations, fps, max(list(all_lengths) + [n_frames]))

    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at [{abs_path}]')

    return out_path


def save_multiple_samples(out_path, file_templates, animations, fps, max_frames, no_dir=False):
    num_samples_in_out_file = 3
    n_samples = animations.shape[0]

    for sample_i in range(0, n_samples, num_samples_in_out_file):
        last_sample_i = min(sample_i + num_samples_in_out_file, n_samples)
        all_sample_save_file = file_templates['all'].format(sample_i, last_sample_i - 1)
        if no_dir and n_samples <= num_samples_in_out_file:
            all_sample_save_path = out_path
        else:
            all_sample_save_path = os.path.join(out_path, all_sample_save_file)
            print(f'saving {os.path.split(out_path)[1]}/{all_sample_save_file}')

        clips = clips_array(animations[sample_i:last_sample_i])
        clips.duration = max_frames / fps

        clips.write_videofile(all_sample_save_path, fps=fps, threads=4, logger=None)

        for clip in clips.clips:
            clip.close()
        clips.close()


def construct_template_variables(unconstrained):
    row_file_template = 'sample{:02d}.mp4'
    all_file_template = 'samples_{:02d}_to_{:02d}.mp4'
    if unconstrained:
        sample_file_template = 'row{:02d}_col{:02d}.mp4'
        sample_print_template = '[{} row #{:02d} column #{:02d} | -> {}]'
        row_file_template = row_file_template.replace('sample', 'row')
        row_print_template = '[{} row #{:02d} | all columns | -> {}]'
        all_file_template = all_file_template.replace('samples', 'rows')
        all_print_template = '[rows {:02d} to {:02d} | -> {}]'
    else:
        sample_file_template = 'sample{:02d}_rep{:02d}.mp4'
        sample_print_template = '["{}" ({:02d}) | Rep #{:02d} | -> {}]'
        row_print_template = '[ "{}" ({:02d}) | all repetitions | -> {}]'
        all_print_template = '[samples {:02d} to {:02d} | all repetitions | -> {}]'

    return sample_print_template, row_print_template, all_print_template, \
           sample_file_template, row_file_template, all_file_template


def load_dataset(args, max_frames, n_frames):
    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              split='test',
                              hml_mode='train' if args.pred_len > 0 else 'text_only',
                              fixed_len=args.pred_len + args.context_len, pred_len=args.pred_len, device=dist_util.dev())
    data.fixed_length = n_frames
    return data


if __name__ == "__main__":
    main()
