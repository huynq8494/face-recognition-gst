import torch
from torchvision import transforms
from PIL import Image
import cv2
import os
import numpy as np
import argparse
from pathlib import Path
import hdbscan
import shutil
from sklearn.metrics import pairwise_distances
import json
import struct
from sklearn.cluster import KMeans
import onnxruntime as ort
import sys

# Add the path for SCRFD imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import SCRFD for face detection
from my_scrfd import SCRFD

class FaceDetector:
    def __init__(self, device='cpu', crop_size=(112, 112),
                 face_det_model="models/scrfd/scrfd_500m_bnkps_shape640x640.onnx",
                 face_rec_model="models/edgeface/edgeface_xs_gamma_06.onnx"):
        self.device = device
        self.crop_size = crop_size

        # Initialize SCRFD for face detection
        self.detector = SCRFD(model_file=face_det_model)
        ctx_id = 0 if device == 'cuda' else -1
        self.detector.prepare(ctx_id, input_size=(640, 640))

        # Initialize ONNX model for face recognition
        if not os.path.exists(face_rec_model):
            raise FileNotFoundError(f"Face recognition model not found: {face_rec_model}")
        self.recognition_session = ort.InferenceSession(face_rec_model)

        self.supported_image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        self.supported_video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
        self.face_data = []

        # Setup transforms for face recognition preprocessing
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def align_face(self, img, kps):
        """Align face using keypoints for better recognition"""
        if kps is None or len(kps) != 5:
            return None

        # Simple face alignment using eye centers
        eye_left = kps[0]
        eye_right = kps[1]

        # Calculate angle
        dY = eye_right[1] - eye_left[1]
        dX = eye_right[0] - eye_left[0]
        angle = np.degrees(np.arctan2(dY, dX))

        # Get rotation matrix
        eyes_center = ((eye_left[0] + eye_right[0]) // 2,
                      (eye_left[1] + eye_right[1]) // 2)
        M = cv2.getRotationMatrix2D(eyes_center, angle, 1)

        # Apply rotation
        aligned = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                                flags=cv2.INTER_CUBIC)
        return aligned

    def crop_face(self, img, bbox, kps=None):
        """Crop and resize face for recognition"""
        x1, y1, x2, y2 = bbox[:4].astype(int)

        # Add padding
        padding = 0
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(img.shape[1], x2 + padding)
        y2 = min(img.shape[0], y2 + padding)

        face = img[y1:y2, x1:x2]

        # Align face if keypoints available
        if kps is not None:
            # Adjust keypoints relative to crop
            adjusted_kps = kps - np.array([x1, y1])
            aligned_face = self.align_face(face, adjusted_kps)
            if aligned_face is not None:
                face = aligned_face

        # Resize to required size
        face = cv2.resize(face, self.crop_size)
        return face

    def detect_faces(self, input_source):
        """Detect faces using SCRFD ONNX model"""
        try:
            # Load image
            if isinstance(input_source, str) or isinstance(input_source, Path):
                img_bgr = cv2.imread(str(input_source))
                if img_bgr is None:
                    img = Image.open(input_source).convert('RGB')
                    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            elif isinstance(input_source, np.ndarray):
                img_bgr = input_source
            else:
                # PIL Image
                img_bgr = cv2.cvtColor(np.array(input_source), cv2.COLOR_RGB2BGR)

            # Detect faces using SCRFD
            bboxes, kpss = self.detector.detect(img_bgr, thresh=0.8, input_size=(640, 640))

            if len(bboxes) == 0:
                return None, None, None

            # Crop faces
            faces = []
            for i in range(len(bboxes)):
                kps = kpss[i] if kpss is not None else None
                face = self.crop_face(img_bgr, bboxes[i], kps)
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face_pil = Image.fromarray(face_rgb)

                # # Write face to image
                # face_filename = f"face_{i}.jpg"
                # face_pil.save(face_filename)

                faces.append(face_pil)

            # quit()

            return bboxes, faces, img_bgr

        except Exception as e:
            print(f"Error processing input: {str(e)}")
            return None, None, None

    def extract_embedding(self, face_image):
        """Extract face embedding using ONNX model"""
        try:
            if isinstance(face_image, np.ndarray):
                face_pil = Image.fromarray(face_image)
            else:
                face_pil = face_image

            # Ensure correct size
            face_pil = face_pil.resize(self.crop_size)

            # Preprocess
            input_tensor = self.transform(face_pil).unsqueeze(0)
            input_numpy = input_tensor.numpy()

            # Run inference
            input_name = self.recognition_session.get_inputs()[0].name
            outputs = self.recognition_session.run(None, {input_name: input_numpy})

            embedding = outputs[0][0]  # Remove batch dimension

            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)

            return embedding

        except Exception as e:
            print(f"Error extracting embedding: {str(e)}")
            return None

    def visualize_detections(self, image_path, bboxes, window_name="Face Detection"):
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Error: Could not load image {image_path}")
            return

        for bbox in bboxes:
            x1, y1, x2, y2, score = bbox[:5].astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{score:.2f}', (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow(f"{window_name} - Press 'ESC' key to exit", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # def extract_and_store_embeddings(self, faces, source_name):
    #     for i, face in enumerate(faces):
    #         if face is not None:
    #             embedding = self.extract_embedding(face)
    #             if embedding is not None:
    #                 self.face_data.append({
    #                     'embedding': embedding,
    #                     'face_image': face,
    #                     'source': source_name,
    #                     'image_inid': i
    #                 })
    def extract_and_store_embeddings(self, faces, source_name, blur_threshold=100.0):
        """
        Extract embeddings and store face data, filtering out blurred faces

        Args:
            faces: List of face images
            source_name: Source identifier for the faces
            blur_threshold: Threshold for blur detection (lower = more blurry)
        """
        for i, face in enumerate(faces):
            if face is not None:
                # Convert PIL Image to numpy array for blur detection
                if isinstance(face, Image.Image):
                    face_np = np.array(face)
                else:
                    face_np = face

                # Convert to grayscale for blur detection
                gray = cv2.cvtColor(face_np, cv2.COLOR_RGB2GRAY)

                # Calculate Laplacian variance (measure of blur)
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

                # Skip blurred faces
                if laplacian_var < blur_threshold:
                    print(f"Skipping blurred face from {source_name} (blur score: {laplacian_var:.2f})")
                    continue

                embedding = self.extract_embedding(face)
                if embedding is not None:
                    self.face_data.append({
                        'embedding': embedding,
                        'face_image': face,
                        'source': source_name,
                        'image_inid': i,
                        'blur_score': laplacian_var  # Store blur score for reference
                    })


    def cluster_faces(self, min_cluster_size=2, min_samples=1):
        if len(self.face_data) < min_cluster_size:
            print(f"Not enough faces for clustering. Found {len(self.face_data)} faces.")
            return None

        embeddings = np.array([data['embedding'] for data in self.face_data])
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                    min_samples=min_samples,
                                    metric='euclidean')
        cluster_labels = clusterer.fit_predict(embeddings)

        for i, label in enumerate(cluster_labels):
            self.face_data[i]['cluster'] = label

        return cluster_labels

    def cluster_faces_kmeans(self, n_clusters=23, random_state=42):
        # n_clusters=8
        n_clusters=23
        random_state=42
        if len(self.face_data) < n_clusters:
            print(f"Not enough faces for {n_clusters} clusters. Found {len(self.face_data)} faces.")
            # Adjust number of clusters to number of faces
            n_clusters = len(self.face_data)
            if n_clusters < 2:
                print("Need at least 2 faces for clustering.")
                return None

        embeddings = np.array([data['embedding'] for data in self.face_data])

        # Initialize and fit KMeans
        kmeans = KMeans(n_clusters=n_clusters,
                    random_state=random_state,
                    n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

        # Store cluster assignments
        for i, label in enumerate(cluster_labels):
            self.face_data[i]['cluster'] = label

        # Calculate and print cluster statistics
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        print(f"K-Means clustering complete: {len(unique_labels)} clusters")
        for label, count in zip(unique_labels, counts):
            print(f"  Cluster {label}: {count} faces")

        return cluster_labels


    def save_face_image(self, face_image, filepath):
        if isinstance(face_image, torch.Tensor):
            face_np = face_image.permute(1, 2, 0).numpy()
            face_np = (face_np * 255).astype(np.uint8)
            face_pil = Image.fromarray(face_np)
        elif isinstance(face_image, np.ndarray):
            face_pil = Image.fromarray(face_image)
        else:
            face_pil = face_image
        face_pil.save(filepath)

    def save_clustered_faces(self, output_dir='clustered_faces'):
        dir_preprocessing = os.path.join(output_dir, "preprocessing")
        dir_representatives = os.path.join(output_dir, "representatives")
        os.makedirs(dir_preprocessing, exist_ok=True)
        os.makedirs(dir_representatives, exist_ok=True)

        cluster_labels = set([data.get('cluster', -1) for data in self.face_data])

        for cluster_id in cluster_labels:
            prefix = 'unclustered' if cluster_id == -1 else f"cluster_{cluster_id}"
            cluster_faces = [data for data in self.face_data if data.get('cluster', -1) == cluster_id]

            for i, face_data in enumerate(cluster_faces):
                face_data['cluster_inid'] = i
                filename = f"{prefix}_{i}_{face_data['source']}_face_{face_data['image_inid']}.jpg"
                filepath = os.path.join(dir_preprocessing, filename)
                self.save_face_image(face_data['face_image'], filepath)

                if i == 0:
                    shutil.copy2(filepath, os.path.join(dir_representatives, f"{prefix}.jpg"))

    def save_detected_faces(self, faces, output_dir, base_filename):
        os.makedirs(output_dir, exist_ok=True)
        for i, face in enumerate(faces):
            if face is not None:
                face_path = os.path.join(output_dir, f"{base_filename}_face_{i+1}.jpg")
                self.save_face_image(face, face_path)

    def perform_clustering(self, min_cluster_size, output_dir):
        if len(self.face_data) > 0:
            print(f"\nClustering {len(self.face_data)} faces...")
            # cluster_labels = self.cluster_faces(min_cluster_size=min_cluster_size)
            cluster_labels = self.cluster_faces_kmeans()
            if cluster_labels is not None:
                unique_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                noise_points = np.sum(cluster_labels == -1)
                print(f"Found {unique_clusters} clusters and {noise_points} noise points")
                self.save_clustered_faces(output_dir)

    def process_single_image(self, image_path, output_dir='output_faces', cluster=False):
        bboxes, faces, img = self.detect_faces(image_path)
        if bboxes is not None:
            print(f"{image_path}: Found {len(faces)} faces")
            base_filename = Path(image_path).stem

            if cluster:
                self.extract_and_store_embeddings(faces, base_filename)
            else:
                self.visualize_detections(image_path, bboxes)
                self.save_detected_faces(faces, output_dir, base_filename)
        else:
            print(f"{image_path}: Found 0 faces")

    def process_single_video(self, video_path, output_dir='output_faces', save_frames=False, cluster=False):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return

        base_filename = Path(video_path).stem
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            bboxes, faces, _ = self.detect_faces(frame)

            if bboxes is not None and len(faces) > 0:
                print(f"{video_path}: Frame {frame_count}: Found {len(faces)} faces")
                source_name = f"{base_filename}_frame_{frame_count}"

                if cluster:
                    self.extract_and_store_embeddings(faces, source_name)
                else:
                    self.save_detected_faces(faces, output_dir, source_name)

        cap.release()

    def apply_cluster_config(self, config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

            for data in self.face_data:
                data['user'] = 'dnc'

            for user, clusters in config.items():
                if user not in ['edit', 'unclustered'] and isinstance(clusters, list):
                    for cluster_id in clusters:
                        for data in self.face_data:
                            if data.get('cluster', -1) == cluster_id:
                                data['user'] = user

            if 'unclustered' in config:
                for user, cluster_inids in config['unclustered'].items():
                    for cluster_inid in cluster_inids:
                        for data in self.face_data:
                            if data.get('cluster', -1) == -1 and data['cluster_inid'] == cluster_inid:
                                data['user'] = user

            if 'edit' in config:
                for cluster_id_str, users_dict in config['edit'].items():
                    cluster_id = int(cluster_id_str)
                    for user, cluster_inids in users_dict.items():
                        for cluster_inid in cluster_inids:
                            for data in self.face_data:
                                if data.get('cluster', -1) == cluster_id and data['cluster_inid'] == cluster_inid:
                                    data['user'] = user

            print(f"Applied cluster configuration from {config_path}")

        except Exception as e:
            print(f"Error reading cluster config: {str(e)}")

    def save_user_faces(self, output_dir='clustered_faces'):
        dir_preprocessing = os.path.join(output_dir, "preprocessing")
        dir_postprocessing = os.path.join(output_dir, "postprocessing")
        dir_embeddings = os.path.join(output_dir, "embeddings")
        os.makedirs(dir_postprocessing, exist_ok=True)
        os.makedirs(dir_embeddings, exist_ok=True)

        user_faces = {}
        for data in self.face_data:
            user = data.get('user', 'None')
            if user not in user_faces:
                user_faces[user] = []
            user_faces[user].append(data)

        for user, faces in user_faces.items():
            user_dir = os.path.join(dir_postprocessing, user)
            os.makedirs(user_dir, exist_ok=True)

            for face_data in faces:
                cluster_id = face_data.get('cluster', -1)
                prefix = 'unclustered' if cluster_id == -1 else f"cluster_{cluster_id}"
                filename = f"{prefix}_{face_data['cluster_inid']}_{face_data['source']}_face_{face_data['image_inid']}.jpg"
                source_path = os.path.join(dir_preprocessing, filename)
                dest_path = os.path.join(user_dir, filename)

                if os.path.exists(source_path):
                    shutil.copy2(source_path, dest_path)

            embeddings = np.array([f['embedding'] for f in faces])

            if user == 'dnc':
                continue

            if len(embeddings) > 3:
                kmeans = KMeans(n_clusters=3, random_state=42)
                kmeans.fit(embeddings)
                final_embeddings = kmeans.cluster_centers_
            else:
                final_embeddings = embeddings

            bin_path = os.path.join(dir_embeddings, f"{user}.bin")
            with open(bin_path, 'wb') as f:
                f.write(struct.pack('H', len(final_embeddings)))
                for embedding in final_embeddings:
                    f.write(struct.pack(f'{len(embedding)}f', *embedding))

        print(f"User faces saved to {dir_postprocessing}/")
        print(f"Embeddings saved to {dir_embeddings}/")

    def process_directory(self, input_dir, output_dir, cluster=False):
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"Error: Input directory {input_dir} does not exist")
            return

        image_files = [f for f in input_path.iterdir() if f.suffix.lower() in self.supported_image_extensions]
        video_files = [f for f in input_path.iterdir() if f.suffix.lower() in self.supported_video_extensions]

        if not image_files and not video_files:
            print(f"No supported image or video files found in {input_dir}")
            return

        print(f"Processing {len(image_files)} images and {len(video_files)} videos from {input_dir}")
        print(f"Output directory: {output_dir}")

        for image_file in image_files:
            self.process_single_image(image_file, output_dir, cluster)

        for video_file in video_files:
            self.process_single_video(str(video_file), output_dir, cluster=cluster)

        if cluster:
            self.perform_clustering(2, output_dir)

        config_path = input("Enter path to postprocessing_cluster.json: ").strip()
        if config_path and os.path.exists(config_path):
            self.apply_cluster_config(config_path)
            self.save_user_faces(output_dir)
        else:
            print(f"Config file not found: {config_path}")

    def is_supported_file(self, file_path):
        suffix = Path(file_path).suffix.lower()
        return suffix in self.supported_image_extensions or suffix in self.supported_video_extensions

def main():
    parser = argparse.ArgumentParser(description='Face Detection with SCRFD and Clustering')
    parser.add_argument('input', help='Input image/video file or directory')
    parser.add_argument('-o', '--output', default='output_faces',
                       help='Output directory for detected faces (default: output_faces)')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu',
                       help='Device to run the model on (default: cpu)')
    parser.add_argument('--face-det-model', default='models/scrfd/scrfd_500m_bnkps_shape640x640.onnx',
                       help='Path to SCRFD face detection ONNX model')
    parser.add_argument('--face-rec-model', default='models/edgeface/edgeface_xs_gamma_06.onnx',
                       help='Path to face recognition ONNX model')
    parser.add_argument('--save-frames', action='store_true',
                       help='Save video frames with detected faces')
    parser.add_argument('--cluster', action='store_true',
                       help='Cluster detected faces using HDBSCAN')
    parser.add_argument('--min-cluster-size', type=int, default=2,
                       help='Minimum cluster size for HDBSCAN (default: 2)')

    args = parser.parse_args()

    detector = FaceDetector(device=args.device,
                          face_det_model=args.face_det_model,
                          face_rec_model=args.face_rec_model)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {args.input} does not exist")
        return

    if input_path.is_file():
        if detector.is_supported_file(args.input):
            if input_path.suffix.lower() in detector.supported_image_extensions:
                detector.process_single_image(args.input, args.output, args.cluster)
            else:
                detector.process_single_video(args.input, args.output, args.save_frames, args.cluster)

            if args.cluster:
                detector.perform_clustering(args.min_cluster_size, args.output)
        else:
            print(f"Error: {args.input} is not a supported image or video format")
    elif input_path.is_dir():
        detector.process_directory(args.input, args.output, args.cluster)
    else:
        print(f"Error: {args.input} is neither a file nor a directory")

if __name__ == "__main__":
    main()
