import os
import shutil
from urllib.parse import unquote, urlparse

import numpy as np
import supervisely as sly
from dataset_tools.convert import unpack_if_archive
from supervisely.io.fs import file_exists, get_file_name, get_file_name_with_ext
from tqdm import tqdm

import src.settings as s


def download_dataset(teamfiles_dir: str) -> str:
    """Use it for large datasets to convert them on the instance"""
    api = sly.Api.from_env()
    team_id = sly.env.team_id()
    storage_dir = sly.app.get_data_dir()

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, str):
        parsed_url = urlparse(s.DOWNLOAD_ORIGINAL_URL)
        file_name_with_ext = os.path.basename(parsed_url.path)
        file_name_with_ext = unquote(file_name_with_ext)

        sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
        local_path = os.path.join(storage_dir, file_name_with_ext)
        teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

        fsize = api.file.get_directory_size(team_id, teamfiles_dir)
        with tqdm(
            desc=f"Downloading '{file_name_with_ext}' to buffer...",
            total=fsize,
            unit="B",
            unit_scale=True,
        ) as pbar:
            api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)
        dataset_path = unpack_if_archive(local_path)

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, dict):
        for file_name_with_ext, url in s.DOWNLOAD_ORIGINAL_URL.items():
            local_path = os.path.join(storage_dir, file_name_with_ext)
            teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

            if not os.path.exists(get_file_name(local_path)):
                fsize = api.file.get_directory_size(team_id, teamfiles_dir)
                with tqdm(
                    desc=f"Downloading '{file_name_with_ext}' to buffer...",
                    total=fsize,
                    unit="B",
                    unit_scale=True,
                ) as pbar:
                    api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)

                sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
                unpack_if_archive(local_path)
            else:
                sly.logger.info(
                    f"Archive '{file_name_with_ext}' was already unpacked to '{os.path.join(storage_dir, get_file_name(file_name_with_ext))}'. Skipping..."
                )

        dataset_path = storage_dir
    return dataset_path


def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    ### Function should read local dataset and upload it to Supervisely project, then return project info.###
    dataset_path = "/home/alex/DATASETS/TODO/DOORS/Segmentation"
    im_folder = "img"
    mask_folder = "mask"
    batch_size = 30

    ds_name_to_splits = {
        "ds1 train": "/home/alex/DATASETS/TODO/DOORS/Segmentation/DS1/DS/T.txt",
        "ds1 val": "/home/alex/DATASETS/TODO/DOORS/Segmentation/DS1/DS/V.txt",
        "ds1 test1": "/home/alex/DATASETS/TODO/DOORS/Segmentation/DS1/DS/Te1.txt",
        "ds1 test2": "/home/alex/DATASETS/TODO/DOORS/Segmentation/DS1/DS/Te2.txt",
        "ds2 train": "/home/alex/DATASETS/TODO/DOORS/Segmentation/DS2/DS/T.txt",
        "ds2 val": "/home/alex/DATASETS/TODO/DOORS/Segmentation/DS2/DS/V.txt",
        "ds2 test1": "/home/alex/DATASETS/TODO/DOORS/Segmentation/DS2/DS/Te1.txt",
        "ds2 test2": "/home/alex/DATASETS/TODO/DOORS/Segmentation/DS2/DS/Te2.txt",
    }

    split_pathes_to_tags = {
        "/home/alex/DATASETS/TODO/DOORS/Segmentation/DS1/DS/T.txt": "/home/alex/DATASETS/TODO/DOORS/Segmentation/DS1/T_30000_b_2022-08-02 11.14.22.txt",
        "/home/alex/DATASETS/TODO/DOORS/Segmentation/DS1/DS/V.txt": "/home/alex/DATASETS/TODO/DOORS/Segmentation/DS1/V_5000_b_2022-08-02 11.15.52.txt",
        "/home/alex/DATASETS/TODO/DOORS/Segmentation/DS1/DS/Te1.txt": "/home/alex/DATASETS/TODO/DOORS/Segmentation/DS1/Te1_5000_b_2022-08-02 11.16.00.txt",
        "/home/alex/DATASETS/TODO/DOORS/Segmentation/DS1/DS/Te2.txt": "/home/alex/DATASETS/TODO/DOORS/Segmentation/DS1/Te2_5000_ub_2022-08-02 11.16.11.txt",
        "/home/alex/DATASETS/TODO/DOORS/Segmentation/DS2/DS/T.txt": "/home/alex/DATASETS/TODO/DOORS/Segmentation/DS2/T_20000_b_2022-09-13 22.39.08.txt",
        "/home/alex/DATASETS/TODO/DOORS/Segmentation/DS2/DS/V.txt": "/home/alex/DATASETS/TODO/DOORS/Segmentation/DS2/V_5000_b_2022-09-13 22.40.10.txt",
        "/home/alex/DATASETS/TODO/DOORS/Segmentation/DS2/DS/Te1.txt": "/home/alex/DATASETS/TODO/DOORS/Segmentation/DS2/Te1_5000_b_2022-09-13 22.40.14.txt",
        "/home/alex/DATASETS/TODO/DOORS/Segmentation/DS2/DS/Te2.txt": "/home/alex/DATASETS/TODO/DOORS/Segmentation/DS2/Te2_5000_ub_2022-09-13 22.40.20.txt",
    }

    def create_ann(image_path):
        labels = []
        tags = []

        # tags.append(ds_tag)

        tags_data = name_to_tags_data.get(get_file_name_with_ext(image_path)).split(" ")

        if sub_ds == "DS1":
            boulder_tag = sly.Tag(boulder_meta, value=float(tags_data[7]))
            scale = sly.Tag(scale_meta, value=float(tags_data[8]))
            as_tag = sly.Tag(as_meta, value=float(tags_data[9]))
            ab = sly.Tag(ab_meta, value=float(tags_data[10]))
            sun = sly.Tag(sun_meta, value=float(tags_data[11]))

            tags.extend([boulder_tag, scale, as_tag, ab, sun])

        else:
            as_tag = sly.Tag(as_meta, value=float(tags_data[9]))
            ab = sly.Tag(ab_meta, value=float(tags_data[10]))
            sun = sly.Tag(sun_meta, value=float(tags_data[11]))

        image_np = sly.imaging.image.read(image_path)[:, :, 0]
        img_height = image_np.shape[0]
        img_wight = image_np.shape[1]

        # if ds_name == "test":
        #     tags.append(test_tag)

        mask_path = image_path.replace(im_folder, mask_folder)

        if file_exists(mask_path):
            mask_np = sly.imaging.image.read(mask_path)[:, :, 0]
            mask = mask_np == 255
            if len(np.unique(mask)) > 1:
                curr_bitmap = sly.Bitmap(mask)
                curr_label = sly.Label(curr_bitmap, boulder)
                labels.append(curr_label)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels, img_tags=tags)

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)

    boulder = sly.ObjClass("boulder", sly.Bitmap)

    # ds1 = sly.TagMeta("ds1", sly.TagValueType.NONE)
    # ds2 = sly.TagMeta("ds2", sly.TagValueType.NONE)

    # test1 = sly.TagMeta("test1", sly.TagValueType.NONE)
    # test2 = sly.TagMeta("test2", sly.TagValueType.NONE)

    # name_to_test = {"Te1": test1, "Te2": test2}

    boulder_meta = sly.TagMeta("boulder id", sly.TagValueType.ANY_NUMBER)
    scale_meta = sly.TagMeta("boulder scale", sly.TagValueType.ANY_NUMBER)
    as_meta = sly.TagMeta("albedo of the surface", sly.TagValueType.ANY_NUMBER)
    ab_meta = sly.TagMeta("albedo of the boulder", sly.TagValueType.ANY_NUMBER)
    sun_meta = sly.TagMeta("intensity of the sun", sly.TagValueType.ANY_NUMBER)

    meta = sly.ProjectMeta(
        obj_classes=[boulder],
        tag_metas=[boulder_meta, scale_meta, as_meta, ab_meta, sun_meta],
    )
    api.project.update_meta(project.id, meta.to_json())

    for ds_name, curr_split_path in ds_name_to_splits.items():
        dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)
        # ds_meta = meta.get_tag_meta(sub_ds.lower())
        # ds_tag = sly.Tag(ds_meta)

        sub_ds = curr_split_path.split("/")[-3]
        images_path = os.path.join(dataset_path, sub_ds, "DS", im_folder)

        with open(curr_split_path) as f:
            content = f.read().split("\n")

        images_names = [im_name.rstrip() for im_name in content if len(im_name) > 0]

        tags_path = split_pathes_to_tags[curr_split_path]

        name_to_tags_data = {}
        with open(tags_path) as f:
            content = f.read().split("\n")

        for idx in range(len(images_names)):
            name_to_tags_data[images_names[idx]] = content[idx].rstrip()

        progress = sly.Progress("Create dataset {}".format(ds_name), len(images_names))

        for images_names_batch in sly.batched(images_names, batch_size=batch_size):
            new_im_names = []
            img_pathes_batch = []
            for im_name in images_names_batch:
                img_pathes_batch.append(os.path.join(images_path, im_name))
                new_im_names.append(sub_ds + "_" + im_name)

            img_infos = api.image.upload_paths(dataset.id, new_im_names, img_pathes_batch)
            img_ids = [im_info.id for im_info in img_infos]

            anns = [create_ann(image_path) for image_path in img_pathes_batch]
            api.annotation.upload_anns(img_ids, anns)

            progress.iters_done_report(len(images_names_batch))

    return project
