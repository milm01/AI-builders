import os
import sys
import argparse
import time
import pandas as pd
from ultralytics import YOLO, RTDETR
from PIL import Image, ImageDraw, ImageFont
import torch

def select_balanced_images(test_img_dir, num_compare, num_classes):
    parent_dir = os.path.dirname(test_img_dir)
    label_dir = os.path.join(parent_dir, "labels")
    
    img_extensions = ('.jpg', '.jpeg', '.png')
    all_images = [f for f in os.listdir(test_img_dir) if f.lower().endswith(img_extensions)]
    
    class_to_images = {i: [] for i in range(num_classes)}
    
    for img_name in all_images:
        base_name = os.path.splitext(img_name)[0]
        label_path = os.path.join(label_dir, f"{base_name}.txt")
        
        classes_in_image = set()
        if os.path.exists(label_path):
            try:
                with open(label_path, 'r') as lf:
                    for line in lf:
                        parts = line.strip().split()
                        if parts:
                            class_id = int(parts[0])
                            if 0 <= class_id < num_classes:
                                classes_in_image.add(class_id)
            except Exception:
                pass
        
        for cid in classes_in_image:
            class_to_images[cid].append(img_name)
            
    selected_images = []
    selected_set = set()
    class_indices = {i: 0 for i in range(num_classes)}
    image_trigger_classes = {} 
    
    attempts = 0
    max_attempts = num_compare * num_classes * 5
    
    while len(selected_images) < num_compare and len(selected_set) < len(all_images) and attempts < max_attempts:
        target_class = len(selected_images) % num_classes
        
        img_list = class_to_images[target_class]
        idx = class_indices[target_class]
        
        found = False
        while idx < len(img_list):
            candidate = img_list[idx]
            class_indices[target_class] += 1
            idx += 1
            if candidate not in selected_set:
                selected_images.append(candidate)
                selected_set.add(candidate)
                image_trigger_classes[candidate] = target_class
                found = True
                break
                
        if not found:
            for candidate in all_images:
                if candidate not in selected_set:
                    selected_images.append(candidate)
                    selected_set.add(candidate)
                    image_trigger_classes[candidate] = -1
                    break
        attempts += 1
        
    return selected_images, image_trigger_classes

def parse_args():
    parser = argparse.ArgumentParser(description="Compare YOLO26 with RT-DETR, YOLOv8, and YOLOv11 baselines.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs to train the baselines.")
    parser.add_argument("--batch", type=int, default=4, help="Batch size for training. Lower if out of memory.")
    parser.add_argument("--workers", type=int, default=0, help="Number of dataloader workers. Set to 0 to save system RAM on Windows.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training and evaluation.")
    parser.add_argument("--retrain", action="store_true", help="Force retrain baselines even if trained checkpoints exist.")
    parser.add_argument("--rtdetr-variant", type=str, default="rtdetr-l.pt", 
                        choices=["rtdetr-l.pt", "rtdetr-x.pt"], 
                        help="RT-DETR model variant to use as baseline.")
    parser.add_argument("--train-yolov8", action="store_true", help="Train/Fine-tune YOLOv8n baseline on the dataset.")
    parser.add_argument("--train-yolov11", action="store_true", help="Train/Fine-tune YOLOv11n baseline on the dataset.")
    parser.add_argument("--num-compare", type=int, default=10, help="Number of sample images to generate visual comparisons for.")
    parser.add_argument("--data", type=str, default="bottle.yolo26", help="Name or absolute path of the dataset folder or data.yaml file.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    
    yolo_path = os.path.join(workspace_dir, "aibuilder", "6th.pt")
    if not os.path.exists(yolo_path):
        alt_yolo_path = os.path.join(workspace_dir, "6th.pt")
        if os.path.exists(alt_yolo_path):
            yolo_path = alt_yolo_path
            
    if os.path.isabs(args.data):
        if os.path.isdir(args.data):
            data_yaml = os.path.join(args.data, "data.yaml")
        else:
            data_yaml = args.data
    else:
        direct_path = os.path.abspath(args.data)
        if os.path.isdir(direct_path):
            data_yaml = os.path.join(direct_path, "data.yaml")
        elif os.path.isfile(direct_path):
            data_yaml = direct_path
        else:
            data_yaml = os.path.join(workspace_dir, "aibuilder", args.data, "data.yaml")
            if not os.path.exists(data_yaml) and os.path.exists(os.path.join(workspace_dir, "aibuilder", args.data)):
                data_yaml = os.path.join(workspace_dir, "aibuilder", args.data)
    
    project_dir = os.path.join(workspace_dir, "runs", "detect")
    
    rtdetr_run_name = f"rtdetr_finetuned_{args.rtdetr_variant.split('.')[0]}"
    rtdetr_best_path = os.path.join(project_dir, rtdetr_run_name, "weights", "best.pt")
    
    yolov8_best_path = os.path.join(project_dir, "yolov8_finetuned", "weights", "best.pt")
    yolov11_best_path = os.path.join(project_dir, "yolov11_finetuned", "weights", "best.pt")
    
    print("=" * 60)
    print("        YOLO26 MULTI-MODEL COMPARISON PIPELINE")
    print("=" * 60)
    print(f"Custom YOLO26 Model:   {yolo_path}")
    print(f"Dataset Config:        {data_yaml}")
    print(f"RT-DETR Baseline:      {args.rtdetr_variant}")
    print(f"Device:                {'CUDA (GPU)' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60)

    if not os.path.exists(yolo_path):
        print(f"Error: YOLO26 weights not found at {yolo_path}.")
        sys.exit(1)
    if not os.path.exists(data_yaml):
        print(f"Error: Dataset yaml config not found at {data_yaml}.")
        sys.exit(1)

    if not os.path.exists(rtdetr_best_path) or args.retrain:
        print(f"\n[STEP 1a/3] Fine-tuning RT-DETR baseline ({args.rtdetr_variant})...")
        base_rtdetr = RTDETR(args.rtdetr_variant)
        rtdetr_batch = min(args.batch, 2)
        print(f"Using capped batch size {rtdetr_batch} for RT-DETR to avoid CUDA OOM.")
        base_rtdetr.train(
            data=data_yaml,
            epochs=args.epochs,
            batch=rtdetr_batch,
            imgsz=args.imgsz,
            device=0 if torch.cuda.is_available() else "cpu",
            project=project_dir,
            name=rtdetr_run_name,
            exist_ok=True,
            workers=args.workers,
            cache=False
        )
        print("RT-DETR fine-tuning complete!")

    if args.train_yolov8 and (not os.path.exists(yolov8_best_path) or args.retrain):
        print(f"\n[STEP 1b/3] Fine-tuning YOLOv8 baseline (yolov8n)...")
        base_yolov8 = YOLO("yolov8n.pt")
        base_yolov8.train(
            data=data_yaml,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=0 if torch.cuda.is_available() else "cpu",
            project=project_dir,
            name="yolov8_finetuned",
            exist_ok=True,
            workers=args.workers,
            cache=False
        )
        print("YOLOv8 fine-tuning complete!")

    if args.train_yolov11 and (not os.path.exists(yolov11_best_path) or args.retrain):
        print(f"\n[STEP 1c/3] Fine-tuning YOLOv11 baseline (yolo11n)...")
        base_yolov11 = YOLO("yolo11n.pt")
        base_yolov11.train(
            data=data_yaml,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=0 if torch.cuda.is_available() else "cpu",
            project=project_dir,
            name="yolov11_finetuned",
            exist_ok=True,
            workers=args.workers,
            cache=False
        )
        print("YOLOv11 fine-tuning complete!")

    print("\n[STEP 2/3] Loading all available models for evaluation...")
    models_to_eval = {
        "YOLO26 (yolo26n)": YOLO(yolo_path)
    }
    
    if os.path.exists(rtdetr_best_path):
        models_to_eval[f"RT-DETR ({args.rtdetr_variant.split('-')[1]})"] = RTDETR(rtdetr_best_path)
    else:
        print("Note: RT-DETR baseline weights not found. Run training first to include in comparison.")
        
    if os.path.exists(yolov8_best_path):
        models_to_eval["YOLOv8 (yolov8n)"] = YOLO(yolov8_best_path)
        
    if os.path.exists(yolov11_best_path):
        models_to_eval["YOLOv11 (yolo11n)"] = YOLO(yolov11_best_path)

    comparison_data = {}
    
    for name, model in models_to_eval.items():
        print(f"\nValidating {name}...")
        metrics = model.val(data=data_yaml, split='val', verbose=False, workers=args.workers)
        info = model.info()
        
        comparison_data[name] = {
            "Precision (P)": metrics.results_dict.get('metrics/precision(B)', 0.0),
            "Recall (R)": metrics.results_dict.get('metrics/recall(B)', 0.0),
            "mAP@50": metrics.results_dict.get('metrics/mAP50(B)', 0.0),
            "mAP@50-95": metrics.results_dict.get('metrics/mAP50-95(B)', 0.0),
            "Latency (ms/img)": metrics.speed.get('inference', 0.0),
            "Parameters": info[1],
            "FLOPs (GFLOPs)": info[3]
        }

    metrics_keys = ["Precision (P)", "Recall (R)", "mAP@50", "mAP@50-95", "Latency (ms/img)", "Parameters", "FLOPs (GFLOPs)"]
    
    df_data = {"Metric": metrics_keys}
    for model_name in models_to_eval.keys():
        df_data[model_name] = [comparison_data[model_name][mk] for mk in metrics_keys]
        
    df = pd.DataFrame(df_data)
    csv_path = os.path.join(workspace_dir, "model_comparison.csv")
    df.to_csv(csv_path, index=False)
    
    print("\n" + "=" * 100)
    print("                                MODEL COMPARISON SUMMARY")
    print("=" * 100)
    
    row_format = "{:<20}" + " | {:<22}" * len(models_to_eval)
    
    print(row_format.format("Metric", *models_to_eval.keys()))
    print("-" * (20 + 25 * len(models_to_eval)))
    
    for mk in metrics_keys:
        row_vals = []
        for model_name in models_to_eval.keys():
            val = comparison_data[model_name][mk]
            if mk in ["Precision (P)", "Recall (R)", "mAP@50", "mAP@50-95"]:
                row_vals.append(f"{val:.4f}")
            elif mk == "Latency (ms/img)":
                row_vals.append(f"{val:.2f} ms")
            elif mk == "Parameters":
                row_vals.append(f"{int(val):,}")
            elif mk == "FLOPs (GFLOPs)":
                row_vals.append(f"{val:.4f}")
        print(row_format.format(mk, *row_vals))
        
    print("=" * 100)
    print(f"Spreadsheet successfully saved to: {csv_path}")

    print("\n[STEP 3/3] Generating visual comparison on sample images...")
    dataset_dir = os.path.dirname(data_yaml)
    test_img_dir = os.path.join(dataset_dir, "test", "images")
    if not os.path.exists(test_img_dir):
        test_img_dir = os.path.join(dataset_dir, "valid", "images")
        
    if os.path.exists(test_img_dir):
        yolo26_key = list(models_to_eval.keys())[0]
        class_names = models_to_eval[yolo26_key].names
        num_classes = len(class_names)
        
        selected_images, trigger_classes = select_balanced_images(test_img_dir, args.num_compare, num_classes)
        
        if selected_images:
            output_visual_dir = os.path.join(workspace_dir, "comparison_outputs")
            os.makedirs(output_visual_dir, exist_ok=True)
            
            for img_name in selected_images:
                img_path = os.path.join(test_img_dir, img_name)
                trigger_id = trigger_classes.get(img_name, -1)
                trigger_name = class_names.get(trigger_id, "General/Mixed")
                
                plotted_images = {}
                for m_name, m_model in models_to_eval.items():
                    pred = m_model(img_path, verbose=False)[0]
                    plotted_bgr = pred.plot()
                    plotted_rgb = Image.fromarray(plotted_bgr[..., ::-1])
                    plotted_images[m_name] = (plotted_rgb, len(pred.boxes))
                
                num_imgs = len(plotted_images)
                first_img = list(plotted_images.values())[0][0]
                w, h = first_img.size
                
                if num_imgs <= 2:
                    total_w = w * num_imgs
                    total_h = h + 40
                    combined = Image.new("RGB", (total_w, total_h), (255, 255, 255))
                    draw = ImageDraw.Draw(combined)
                    
                    for i, (m_name, (img, num_boxes)) in enumerate(plotted_images.items()):
                        combined.paste(img, (i * w, 40))
                        draw.text((i * w + 10, 10), f"{m_name} - Detections: {num_boxes}", fill=(0, 0, 0))
                else:
                    total_w = w * 2
                    total_h = (h + 40) * 2
                    combined = Image.new("RGB", (total_w, total_h), (255, 255, 255))
                    draw = ImageDraw.Draw(combined)
                    
                    positions = [
                        (0, 0),
                        (w, 0),
                        (0, h + 40),
                        (w, h + 40)
                    ]
                    
                    for i, (m_name, (img, num_boxes)) in enumerate(plotted_images.items()):
                        if i >= 4:
                            break
                        x, y = positions[i]
                        combined.paste(img, (x, y + 40))
                        draw.text((x + 10, y + 10), f"{m_name} - Detections: {num_boxes}", fill=(0, 0, 0))
                
                save_path = os.path.join(output_visual_dir, f"compare_{img_name}")
                combined.save(save_path)
                print(f"Saved visual comparison grid for {img_name} (Target Class: {trigger_name}) to {save_path}")
        else:
            print("No test/validation images found to generate visual comparisons.")
    else:
        print("Could not find test or validation image directory for visual comparison.")

if __name__ == "__main__":
    main()
