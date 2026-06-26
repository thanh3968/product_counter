# Product Counting On A High Speed Conveyor

Project Python/OpenCV mô phỏng hệ thống đếm sản phẩm trên băng chuyền tốc độ cao, tương tự ý tưởng trong video YouTube: phát hiện vật thể, tracking centroid, và đếm khi sản phẩm đi qua vạch kiểm tra.

## Mở bằng VS Code

1. Mở folder này trong VS Code:
   `outputs/product-counter-vscode`
2. Tạo môi trường ảo nếu muốn:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
3. Cài thư viện:
   ```powershell
   pip install -r requirements.txt
   ```

## Chạy demo

Tạo video băng chuyền mẫu:

```powershell
python src/generate_demo.py
```

Chạy bộ đếm sản phẩm:

```powershell
python src/product_counter.py --video demo/conveyor_demo.mp4 --output demo/conveyor_counted.mp4
```

Kết quả:

- `demo/conveyor_demo.mp4`: video demo gốc.
- `demo/conveyor_counted.mp4`: video đã annotate bounding box, vạch đếm, ID tracking và tổng số sản phẩm.

Trong VS Code, bạn cũng có thể dùng tab Run and Debug:

- `Generate demo video`
- `Run product counter`

## Thay bằng video thật

```powershell
python src/product_counter.py --video path\to\your_video.mp4 --output demo\your_counted.mp4
```

Các tham số hữu ích:

```powershell
python src/product_counter.py --video demo/conveyor_demo.mp4 --line-x 620 --min-area 900 --max-distance 85
```

- `--line-x`: vị trí vạch đếm theo trục X.
- `--min-area`: bỏ qua nhiễu nhỏ.
- `--max-distance`: khoảng cách tối đa để nối object giữa 2 frame.
- `--display`: hiển thị cửa sổ realtime khi chạy.

