#!/usr/bin/env python3
"""
Real-time Octo Input Viewer
Monitors the debug images saved by octo_bridge.py
"""

import cv2
import time
import os
import glob
import argparse

def monitor_octo_inputs(watch_dir="/tmp", refresh_rate=1.0):
    """
    Monitor and display Octo model input images in real-time
    
    Args:
        watch_dir: Directory to monitor for debug images
        refresh_rate: How often to check for new images (seconds)
    """
    print(f"🔍 Monitoring Octo model inputs in {watch_dir}")
    print("Press 'q' to quit, 's' to save current image")
    
    last_files = set()
    
    while True:
        try:
            # Find all debug images
            pattern = os.path.join(watch_dir, "debug_octo_input_*.jpg")
            current_files = set(glob.glob(pattern))
            
            # Check for new files
            new_files = current_files - last_files
            
            if new_files:
                # Sort by timestamp (newest first)
                latest_file = max(new_files, key=os.path.getctime)
                
                print(f"📸 New input image: {os.path.basename(latest_file)}")
                
                # Load and display image
                img = cv2.imread(latest_file)
                if img is not None:
                    # Resize for better viewing
                    height, width = img.shape[:2]
                    if width > 800:
                        scale = 800 / width
                        new_width = 800
                        new_height = int(height * scale)
                        img = cv2.resize(img, (new_width, new_height))
                    
                    # Add filename info
                    filename = os.path.basename(latest_file)
                    timestamp = filename.split('_')[-1].replace('.jpg', '')
                    
                    cv2.putText(img, f"File: {filename}", (10, img.shape[0] - 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    
                    cv2.putText(img, f"Timestamp: {timestamp}", (10, img.shape[0] - 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    
                    cv2.imshow('Octo Model Input (Real-time)', img)
                
                last_files = current_files
            
            # Handle key presses
            key = cv2.waitKey(int(refresh_rate * 1000)) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and 'img' in locals():
                save_path = f"saved_octo_input_{int(time.time())}.jpg"
                cv2.imwrite(save_path, img)
                print(f"💾 Saved current image to: {save_path}")
            elif key == ord('c'):
                # Clear old debug files
                for f in glob.glob(pattern):
                    try:
                        os.remove(f)
                        print(f"🗑️ Removed {f}")
                    except:
                        pass
                last_files.clear()
                print("🧹 Cleared old debug files")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(refresh_rate)
    
    cv2.destroyAllWindows()
    print("👋 Monitoring stopped")

def main():
    parser = argparse.ArgumentParser(description='Monitor Octo model input images')
    parser.add_argument('--watch_dir', default='/tmp', 
                       help='Directory to monitor for debug images')
    parser.add_argument('--refresh_rate', type=float, default=0.5,
                       help='Refresh rate in seconds')
    parser.add_argument('--cleanup', action='store_true',
                       help='Clean up old debug files on startup')
    
    args = parser.parse_args()
    
    if args.cleanup:
        pattern = os.path.join(args.watch_dir, "debug_octo_input_*.jpg")
        old_files = glob.glob(pattern)
        for f in old_files:
            try:
                os.remove(f)
                print(f"🗑️ Cleaned up: {f}")
            except:
                pass
    
    monitor_octo_inputs(args.watch_dir, args.refresh_rate)

if __name__ == '__main__':
    main()
