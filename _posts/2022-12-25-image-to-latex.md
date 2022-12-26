---
title: "Image2Latex: Từ hình ảnh đến công thức Latex"
author: tuanio
date: 2022-12-25 02:59:00 +/-0084
categories: [knowledge]
tags: [machine learning, computer vision, image captioning, im2latex-100k dataset, encoder-decoder model, supervised learning, latex]
toc: true
comments: true
published: false
math: true
---

### Nội dung
- [1. Giới thiệu bài toán Image2Latex](#-introduction)
- [2. Kiến trúc AutoRec](#-kien-truc)
- [3. Thực nghiệm với bộ dữ liệu MovieLens](#-thuc-nghiem)
    - [3.1 Chuẩn bị dữ liệu](#-du-lieu)
    - [3.2 Thiết kế mô hình AutoRec](#-thiet-ke)
- [4. Tổng kết](#-tong-ket)
- [5. Tham khảo](#-tham-khao)


<a name="-review"></a>
# 1. Giới thiệu bài toán Image2Latex và các nghiên cứu liên quan

<a name="-introduction"></a>
## 1.1. Giới thiệu

Ta chứng kiến được càng ngày càng có nhiều người có nhu cầu trong việc viết và sử dụng $\text{LaTeX}$, nhưng việc đọc một tài liệu hay hình ảnh có công thức latex mà chưa biết ký hiệu $\text{LaTeX}$ của nó khiến cho người đọc và người sử dụng $\text{LaTeX}$ gặp khó khăn. Trong bài viết này, tôi sẽ giới thiệu bài toán này là gì, có các nghiên cứu ra sao và trình bày các cách kết hợp các kiến trúc Encoder khác nhau để cấu tạo lên mô hình Encoder-Decoder trong việc giải quyết bài toán Image to Latex. Trong số các kết quả, sự kết hợp giữa Convolution Feature Encoder và BiLSTM Row Encoder đạt được kết quả tốt nhất trên BLEU4 là $77\%$ trong bộ dữ liệu `IMG2LATEX-100k`. 

Bài toán hình ảnh sang $\text{LaTeX}$ (Image to Latex) nhằm mục đích giải quyết vấn đề chuyển đổi hình ảnh chứa các công thức toán học $\text{LaTeX}$ trong sách, hình ảnh sang chuỗi $\text{LaTeX}$ thực tế để tiện cho việc biên soạn, chỉnh sửa hay chỉ đơn giản là tò mò xem công thức ở trong ảnh kia nó có tên $\text{LaTeX}$ là gì để có thể tìm kiếm các tài liệu nghiên cứu liên quan đến ký tự đó.

Đầu vào của bài toán này là một hình ảnh chứa chuỗi $\text{LaTeX}$, đầu ra sẽ là một chuỗi $\text{LaTeX}$, Hình 1 mô tả sơ đồ bài toán.

<p>
    <img src="/assets/image2latex/img_to_latex_diagram.png" alt="img2latex-diagram"/>
    <em>Hình 1. Sơ đồ bài toán Image2Latex. Đầu vào sẽ là một ảnh chứa công thức Latex, và đầu ra sẽ mô tả công thức đó dưới dạng chuỗi.</em>
</p>

Bài toán này thoạt nhìn thì khá tương tự với bài toán Chú thích hình ảnh (Image Captioning) [[1]](#-image_captioning_paperwithcode), Nhận dạng giọng nói (Automatic Speech Recognition - ASR) [[2]](#-asr_paperwithcode) hay Nhận dạng chữ quang học (Optical Character Recognition - OCR) [[3]](#-ocr_paperwithcode). Tuy nhiên, bài toán Image to Latex có những sự khác biệt nhất định với ba bài toán trên. Hình ảnh đầu vào của bài toán chỉ đơn giản là ảnh trắng đen, không nhiều màu sắc và thông tin như bài toán Image Captioning, nên mô hình sẽ học khó hơn về độ phức tạp. Các công thức $\text{LaTeX}$ có cách hiển thị và kết xuất khác nhau (ví dụ công thức $\sum_{i=1}^N i^2$ trong Latex sẽ được hiển thị là `\sum_{i=1}^N i^2`, nên sẽ dẫn đến chồng chéo các công thức, ký hiệu Latex lên nhau, trong khi đó bài toán OCR lại chỉ nhận dạng các ký tự theo nhau liên tục, đơn điệu, không bị chồng chéo lên nhau. Thêm nữa, mặc dù bài toán ASR có dữ liệu gốc là dạng sóng của âm thanh, nhưng khi chuyển về dạng Mel Spectrogram thì vẫn sẽ được hiển thị như một ảnh, nhưng đầu ra cũng sẽ là các ký tự đơn điệu với nhau, không chồng chéo lên nhau, vẫn sẽ khác so với bài toán Image to Latex. Vì thế, các kiến trúc mô hình, các phương pháp của Image Captioning, OCR hay ASR không thể sử dụng ngay cho bài toán Image to Latex được. Hình 2 mô tả các nội dung trên.

<p>
    <img src="/assets/image2latex/summary_problems.png" alt="summary_problems"/>
    <em>Hình 2. Image2Latex thoạt nhìn thì tưởng giống, nhưng nhìn kỹ thì lại khác so với 3 bài toán còn lại.</em>
</p>

<a name="-introduction"></a>
## 1.2. Các nghiên cứu liên quan

Bài toán này đã được nghiên cứu một thời gian. Nhóm tác giả [[1]](#-image_to_latex_standford) sử dụng kiến trúc Encoder-Decoder (Encoder là các lớp CNN kết hợp với Max Pooling và Decoder là LSTM 1 lớp) với Luong Attention đạt kết quả $75\%$ trên $\text{BLEU}$ và $35\%$ trên Exact Match (EM).

Nhóm tác giả [[2]](#-image_to_latex_havard) trình bày mô hình $\textit{What You Get Is What You See}$ (WYGIWYS), là một trình dịch ngược từ hình ảnh, cũng sử dụng Encoder-Decoder, nhưng Encoder hiện tại là sự kết hợp của CNN, Max Pooling, Batch Normalization [[6]](#-batch_norm) và thêm một Bi-LSTM 1 lớp (gọi là Row Encoder) chạy trên từng hàng của dữ liệu ảnh được trích xuất từ Convolution Encoder, và Decoder là 1 lớp LSTM. Kết quả đạt được của kiến trúc này là $87.73\%$ trên BLEU và $77.46\%$ trên EM.

Cả hai bài nghiên cứu trên đều được thực nghiệm trên bộ dữ liệu $\text{IM2LATEX-100K}$ [[7]](#-im2latex_100k), nhóm cũng tham khảo hai bài nghiên cứu trên cho việc thực nghiệm.

<a name="-method"></a>
# 2. Các phương pháp có thể tiếp cận

Trong bài blog này, tôi sử dụng kiến trúc mô hình Encoder-Decoder với Attention, bao gồm Encoder là các kiến trúc được cấu thành từ CNN làm trung tâm, Decoder là LSTM 1 lớp và Attention sử dụng cơ chế chú ý để kết nối giữa Encoder và Decoder nhằm gia tăng khả năng học ngữ cảnh của mô hình.

<a name="-encoder"></a>
## 2.1. Encoder

<a name="-conv-encoder"></a>
### 2.1.1 Convolution Encoder

Trong bài nghiên cứu [[1]](#-image_to_latex_standford), nhóm tác giả đã đề xuất một cấu trúc kết hợp các lớp CNN lại với nhau để tạo thành Encoder cho mô hình nhằm giải quyết bài toán Image to Latex. Hình dưới đây mô tả kiến trúc của nhóm tác giả. Tổng quan, kiến trúc này bao gồm 6 lớp CNN đều có kernel size là 3x3, stride 1 và 2 lớp Max Pooling. Hình 3 mô tả kiến trúc này.

<p>
    <img src="/assets/image2latex/conv_encoder.png" alt="conv_encoder"/>
    <em>Hình 3. Kiến trúc của Convolution Encoder</em>
</p>

Kích cỡ đầu vào là $C\times W\times H$, sau khi đưa qua mô hình sẽ trở thành ma trận $V$ có kích thước $512\times W' \times H'$ (gọi là feature grid).

Ngoài ra, bên cạnh Convolution Encoder như trên, nhóm phát triển thêm một phiên bản kết hợp với Batch Normalization [[6]](#-batch_norm) vào sau các lớp CNN thứ 2, 3, 5 và 6. Batch Normalization sẽ giúp mô hình được huấn luyện nhanh và ổn định hơn bằng cách chuẩn hóa theo từng mini-batch dữ liệu đầu vào của mỗi lớp, sẽ giúp giảm được Covariance Shift, khiến phân phối dữ liệu đều nhau và thêm một ít chỉnh hóa cho mô hình tránh bị overfit.

<a name="-conv-row-encoder"></a>
### 2.2.3 Convolution Row Encoder

Theo tự nhiên, ta có thể hiểu rằng các ký hiệu $\text{LaTeX}$ sẽ được viết theo thứ tự từ trái sang phải. Vì thế để có thể học được ngữ cảnh "\textit{trái sang phải}" này, nhóm tác giả [[2]](#-image_to_latex_havard) kết hợp thêm kiến trúc của mạng neuron hồi tiếp (RNN) vào sau feature grid được trích xuất từ các lớp CNN (Convolution Encoder).

Cụ thể, kiến trúc Convolution Row Encoder được kết hợp từ Convolution Feature Encoder và Bi-LSTM Row Encoder. Mà Convolution Feature Encoder sẽ làm việc trên ảnh đầu vào, còn Row Encoder (LSTM 1 lớp 2 chiều có kích thước lớp ẩn là 256) sẽ nhận từng hàng của đầu ra của Convolution Feature Encoder để học và đưa ra một ma trận $V$ có kích cỡ $512 \times W' \times H'$. Hình 4 mô tả kiến trúc của Convolution Row Encoder.

<p>
    <img src="/assets/image2latex/conv_row_encoder.png" alt="conv_row_encoder"/>
    <em>Hình 4. Kiến trúc của Convolution Row Encoder</em>
</p>

<a name="-resnet-18"></a>
### 2.3.3 ResNet-18

Sự phát triển của mô hình ResNet [[8]](#-resnet) giúp làm kiến trúc nền đằng sau các bài toán Computer Vision hiện nay như Image Recognition, Object Detection, GAN, ... [[8]](#-resnet) [[10]](#-resnet_object_detection) [[9]](#-resnet_gan). Kiến trúc ResNet chủ yếu là sự kết hợp của các lớp CNN, Max Pooling, Batch Normalization và activation ReLU. Hình 5 mô tả kiến trúc tổng quát của ResNet-18, là kiến trúc nhóm sử dụng trong bài này.

Batch Normalization sẽ được gắn theo sau mỗi lớp Conv2d, bên cạnh đó, kẹp giữa các lớp Conv2d đôi sẽ là activation ReLU. Các skip connection dạng nối liên tục sẽ cộng trực tiếp giá trị gốc vào lớp đó, nghĩa là $\mathrm{g(x)} = \mathrm{x} + \mathrm{f(x)}$, còn dạng nét đứt sẽ cộng giá trị từ một lớp Conv2d thứ 3 (nhằm downsample (#-downsample}), nghĩa là $\mathrm{g(x)} = \mathrm{Conv2d(x)} + \mathrm{f(x)}$. Trong đó $\mathrm{f(x)}$ là giá trị của khối hiện tại sau khi đưa qua đầu vào, $\mathrm{x}$ là giá trị đầu vào, còn $\mathrm{g(x)}$ là kết quả đầu ra của lớp đó. Đầu ra của ResNet-18 sẽ là ma trận $V$ có kích thước $512 \times W' \times H'$.

Ngoài ra, nhóm kết hợp thêm một phiên bản mô hình BiLSTM Row Encoder vào Feature Grid của ResNet-18.

<p>
    <img src="/assets/image2latex/resnet18.png" alt="resnet-18"/>
    <em>Hình 5. Kiến trúc của ResNet-18</em>
</p>

<a name="-decoder-attention"></a>
# 2.2. Decoder & Attention

<a name="-decoder"></a>
## 2.2.1. Decoder

Trong bài này, nhóm chỉ sử dụng một kiểu decoder duy nhất là LSTM 1 lớp với kích thước lớp ẩn là 512. Decoder này sẽ nhận giá trị đầu vào là feature grid $V$ có kích thước $C' \times W' \times H'$, kết hợp với Luong-Attention để lần lượt đưa ra các ký hiệu $\text{LaTeX}$.

Cụ thể hơn, tại mỗi bước thời gian $t$, đầu vào của LSTM decoder sẽ là sự kết hợp giữa vector word embedding của ký hiệu Latex $y_t$ và vector ngữ cảnh attention $c_t$, và vector trạng thái ẩn $h_{t - 1}$. Mà $c_t$ sẽ được tính từ vector trạng thái ẩn trước đó của decoder $h_{t - 1}$ và feature grid của encoder $V$. Sau cùng LSTM sẽ xuất ra một phân phối xác suất của các ký hiệu Latex là đầu ra của bước thời gian $t$ của decoder.

\begin{align}
    c_t & = \mathrm{Attention}(h_{t-1}, V) \\
    e_t & = \mathrm{Embedding}(y_t) \\
    o_t, h_t & = \mathrm{LSTM}(h_{t - 1}, [c_t, e_t]) \\
    p(y_{t + 1} | y_1, \cdots, y_t) & = \mathrm{Softmax}(W_o \cdot o_t + b_o)
\end{align}

Khi bắt đầu, $h_0$ sẽ được khởi tạo bằng cách tổng theo $H'$ và $W'$ trên feature grid $V$ của encoder.

\begin{align}
    h_0 = \mathrm{tanh}(W_h \cdot (\frac{1}{W' \times H'} \sum_{i=1}^{W' \times H'} v_i) + b_h), \quad v_i \in V
\end{align}

Trọng số ban đầu của lớp Embedding và LSTM được khởi tạo theo Orthogonal [[13]](#-orthogonal).

<a name="-attention"></a>
## 2.2.2. Attention

Cơ chế Attention nhóm sử dụng trong bài là Luong Attention [[12]](#-luong_attention).

\begin{align}
    \mathrm{attn} & =\mathrm{tanh}((W_{h} h_{t - 1} + b_{h}) + (W_{V} V + b_{V})) \\ 
    \alpha^t & = \mathrm{Softmax}(W_{a}\mathrm{attn} + b_{a}) \\
    c_t & = \sum_{i=1}^{W' \times H'} \alpha_i^t v_i, \quad v_i \in V
\end{align}

<a name="-experiment-result"></a>
# 3. Thực nghiệm & Kết quả 
 <!-- \label{experiment_and_result} -->

<a name="-experiment-setup"></a>
## 3.1. Thiết lập thực nghiệm

<a name="-data"></a>
### 3.1.1. Dữ liệu

Nhóm sử dụng tập dữ liệu giống với hai tác giả đã giới thiệu trong phần các nghiên cứu liên quan cho mục đích so sánh. Bộ dữ liệu tên là `im2latex-100k` [[14]](#-im2latex_100k_dataset) có tổng cộng $100000$ cặp dữ liệu (ảnh, latex) được chia thành 3 tập: train (dùng cho việc huấn luyện mô hình), test (dùng cho việc đánh giá trong phần \ref{result}) và valid (dùng như tập test nhưng đánh giá trong quá trình huấn luyện) với tỉ lệ lần lượt là ($80 \%$, $9 \%$ và $11 \%$).

<a name="-preprocessing"></a>
### 3.1.2. Tiền xử lý dữ liệu

Đối với ảnh, nhóm chuyển dữ liệu của các điểm ảnh về khoảng số $[-1, 1]$ để dữ liệu có một khoảng giá trị đồng nhất, giúp mô hình có thể học tốt hơn.

Đối với Latex, nhóm thêm ký tự `<s>` và `<e>` vào đầu và cuối mỗi câu latex đã được tokenize để biểu thị cho mô hình có thể biết được vị trí đầu và cuối mỗi câu Latex. Khi đưa câu này vào mô hình, sẽ tạm thời bỏ ký hiệu `<e>` ra, và khi đưa giá trị đầu ra của mô hình cho việc tính giá trị mất mát, sẽ để lại `<e>` nhưng bỏ ký tự `<s>`.

Sau khi tokenize trên toàn bộ các câu Latex của tập dữ liệu huấn luyện, nhóm tổng hợp được 512 ký hiệu Latex khác nhau, đây cũng là số lớp đầu ra của mô hình.

<a name="-metrics"></a>
### 3.1.3. Phương pháp đánh giá

Nhóm sử dụng 3 phương pháp đánh giá: BLEU, Exact Match (EM) và Edit distance.

- BLEU là một phương pháp thường được sử dụng trong các bài toán dịch máy, tóm tắt văn bản, nhận dạng giọng nói hay sinh nhãn ảnh (image captioning). Cách tính của BLEU là đếm số n-gram khớp nhau giữa câu được dự đoán và câu thực tế, sau đó chia số lượng ký tự của câu dự đoán. BLEU có khoảng giá trị $[0, 1]$, càng cao càng tốt.

- EM được tính bằng cách lấy số lượng ký hiệu trong câu dự đoán giống với trong câu thực tế chia cho số lượng ký hiệu latex có trong câu, khoảng giá trị $[0, 1]$, càng cao càng tốt.
    \item Edit distance là phép đo số lượng thay đổi chúng ta phải thực hiện đối với một chuỗi để biến đổi nó thành chuỗi mà chúng ta đang so sánh với, các phép đo bao gồm: thêm, xóa, sửa, khoảng giá trị $[0, 1]$, càng thấp càng tốt.

<a name="-hyperparameters"></a>
### 3.1.4. Mô hình & tham số

Để kiểm tra kết quả và khả năng của các mô hình, nhóm có tổng hợp các phương pháp trên để đem đi so sánh. Cụ thể các mô hình sẽ có Encoder của một trong số Convolution Encoder, Convolution Encoder + Batch Normalization, Convolution Row Encoder, ResNet-18, ResNet-18 Row Encoder (ResNet-18 + BiLSTM Row Encoder). Đối với $V$, số channel đầu ra luôn luôn là 512 (BiLSTM Row Encoder có kích thước lớp ẩn là 256, sau khi đi qua BiLSTM sẽ trở thành 512).

Decoder luôn luôn là LSTM 1 lớp có 512 node của lớp ẩn. 

<a name="-training-parameters"></a>
### 3.1.5. Thông số huấn luyện

Nhóm sử dụng optimizer AdamW [[15]](#-adamw) với learning rate $\eta = 0.002 * \sqrt{\text{batch size}}$, kết hợp thêm OneCycleLR [[16]](#-onecyclelr) để lập lịch cho learning rate với tỉ lệ bước để learning rate lên vị trí cao nhất là $30 \%$ tổng số bước huấn luyện.

Do sự khác nhau giữa kích thước mô hình và phần cứng hạn chế (Kaggle 1 GPU P100), nhóm sử dụng các kích thước batch size khác nhau cho các mô hình (1, 2, 8, ... sao cho vừa GPU), nhưng nhóm kết hợp thêm kỹ thuật Accumulate Gradient [[17]](#-accumulate_gradient) để tổng hợp các batch size nhỏ nhiều lần thành batch size 64, nên có thể nói các mô hình được huấn luyện trên batch size là 64.

Số epoch tối đa là 10 cho các mô hình, tuy nhiên nhóm huấn luyện ResNet-18 và ResNet-18 Row Encoder lâu hơn, có epoch tối đa là 20.

<a name="-miscellaneous"></a>
### 3.1.6. Thông tin khác

Để cải thiện kết quả dự đoán, nhóm sử dụng Beam Search [[18]](#-beam_search) trên mô hình với Beam Width là 5. Source code của nhóm có thể lấy ở GitHub <a href="https://github.com/tuanio/image2latex" target="_blank">tuanio/image2latex</a>.

<a name="-result"></a>
## 3.2. Kết quả
 <!-- \label{result} -->

Train loss (hình 6) cho thấy Convolution Encoder và Convolution Encoder với Batch Norm có mức độ dao động cao nhất, 3 mô hình còn lại có giá trị loss theo thời gian nhìn ổn định hơn (ResNet-18 Row Encoder đôi khi vẫn có dao động). Ở validation edit distance (hình 9) hay validation exact match (hình 10) chỉ có duy nhất Convolution Row Encoder có sự thay đổi trong giá trị, trong khi các mô hình khác không có sự thay đổi theo thời gian, còn ở validation bleu (hình 8) thì tất cả mô hình đều thay đổi, chỉ có Convolution Row Encoder có sự thay đổi rõ ràng nhất, kết hợp thêm kết quả từ bảng 1 chứng tỏ Convolution Row Encoder học ổn nhất trong các mô hình. Bảng \ref{sample_predict} trình bày một số mẫu câu latex được dự đoán ra.

<p>
    <img src="/assets/image2latex/train_loss.jpg" alt="train_loss" style="width: 500px;" />
    <em>Hình 6. Train loss theo thời gian của các mô hình</em>
</p>

<p>
    <img src="/assets/image2latex/val_loss.jpg" alt="val_loss"/>
    <em>Hình 7. Validation loss theo thời gian của các mô hình</em>
</p>

<p>
    <img src="/assets/image2latex/val_bleu4.jpg" alt="val_bleu4"/>
    <em>Hình 8. Validation BLEU4 theo thời gian của các mô hình</em>
</p>

<p>
    <img src="/assets/image2latex/val_edit_distance.jpg" alt="val_edit_distance"/>
    <em>Hình 9. Validation Edit Distance theo thời gian của các mô hình</em>
</p>

<p>
    <img src="/assets/image2latex/val_exact_match.jpg" alt="val_exact_match"/>
    <em>Hình 10. Validation EM theo thời gian của các mô hình</em>
</p>

| Mô hình | BLEU4 (\%) | EM (\%) | Edit distance (\%) |
| --- | --- | --- | --- |
| Convolution Encoder | | $11.18$ | $0$ | $100$ |
| $\quad$ + Batch Normalization | $14.3$ | $0$ | $100$ |
| **Convolution Row Encoder** | $\textbf{77}$ | $\textbf{16.52}$ | $\textbf{31.23}$ |
| ResNet-18 | $25.54$ | $0.02$ | $100$ |
| ResNet-18 Row Encoder | $18.5$ | $0$ | $100$ |
| G. Genthial và cộng sự [[1]](#-image_to_latex_standford}  | $78$ | $35$ | $24$ |
| Y. Deng và cộng sự [[2]](#-image_to_latex_havard} | \textbf{$87.73$} | \textbf{$77.46$} | - |
Bảng 1. Bảng so sánh giá trị BLEU4, EM và Edit distance của nhóm với hai bài nghiên cứu trong phần \ref{related_work}. Kết quả tốt nhất của nhóm và kết quả tốt nhất của các nhóm khác được in đậm. Các giá trị trong bảng đã được $\times 100$.

Ngoài ra, kết quả của bảng 1 cho ta thấy được:

- Việc dùng một BiLSTM cho Encoder đem lại kết quả tốt khi kết hợp với Convolution Feature Encoder (Kết quả có sự thay đổi rõ rệt từ Convolution Encoder, Convolution Encoder + Batch Normalization sang Convolution Row Encoder).

- Batch Normalization cải thiện kết quả đánh giá và giúp quá trình huấn luyện ổn định hơn (dù chỉ một ít) (hình \ref{train_loss}).

- Đối với ResNet-18, mô hình này có sự kết hợp các lớp CNN sâu (18 lớp), sẽ khiến Feature Grid $V$ đầu ra có kích thước của $W' \times H'$ nhỏ, vì thế số hàng có thể học sẽ ít, khiến BiLSTM không thực sự phát huy tác dụng.

- Tuy nhiên mô hình Convolution Encoder hay Convolution Encoder + Batch Normalization có số lớp CNN nhỏ (không sâu) như ResNet-18, nhưng cũng không có kết quả tốt, chứng tỏ cần phải cân bằng giữa mức độ sâu của các lớp CNN và Batch Normalization.

<!-- \begin{table}[htbp]
\caption{Bảng trình bày một số câu dự đoán từ mô hình Convolution Row Encoder.}
\begin{center}
\begin{tabular}{c c}
\textbf{Thực tế} & \textbf{Dự đoán}\\
\hline \\
    \includegraphics[scale=0.31]{img/latex_1_truth.png} & \includegraphics[scale=0.31]{img/latex_1_predict.png} \\ 

    \includegraphics[scale=0.31]{img/latex_2_truth.png} & \includegraphics[scale=0.31]{img/latex_2_predict.png} \\ 

    \includegraphics[scale=0.31]{img/latex_3_truth.png} & \includegraphics[scale=0.31]{img/latex_3_predict.png} \\ 

    \includegraphics[scale=0.31]{img/latex_4_truth.png} & \includegraphics[scale=0.31]{img/latex_4_predict.png} \\

    \includegraphics[scale=0.31]{img/latex_5_truth.png} & \includegraphics[scale=0.31]{img/latex_5_predict.png} \\ \\
\hline
\end{tabular}
\label{sample_predict}
\end{center}
\end{table} -->

<a name="-conclusion"></a>
# 4. Kết luận
<!-- \label{conclusion} -->

Trong bài nghiên cứu này, nhóm đã trình bày về các cách kết hợp với nhau để tạo lên mô hình Encoder-Decoder dành cho bài toán Image to Latex. Kết quả tốt nhất đạt được từ Convolution Row Encoder có BLEU là $77\%$, EM $16.52 \%$ và Edit distance $31.23 \%$ trên tập dữ liệu IM2LATEX-100K, chứng tỏ được rằng thêm một BiLSTM vào encoder giúp học ngữ cảnh tốt hơn. Bên cạnh đó, thực nghiệm cũng cho thấy mô hình ResNet-18 có số lượng lớp quá sâu có thể không thích hợp với bài toán Image to Latex, khi mà thông tin từ ảnh chỉ là trắng đen và cần phải cân bằng giữa số lượng các lớp CNN và Batch Normalization để mô hình có khả năng tổng quát hóa dữ liệu hơn.


<a name="-reference"></a>
# 5. Tham khảo

<a name="-image_to_latex_standford"></a>
[1] G. Genthial và R. Sauvestre, "Image to Latex". <a href="http://cs231n.stanford.edu/reports/2017/pdfs/815.pdf">http://cs231n.stanford.edu/reports/2017/pdfs/815.pdf</a> (2017).

<a name="-image_to_latex_havard"></a>
[2] Y. Deng, A. Kanervisto và A. M. Rush, "What You Get Is What You See: A Visual Markup Decompilerd". <a href="http://arxiv.org/abs/1609.04938">http://arxiv.org/abs/1609.04938</a> (2016).

<a name="-image_captioning_paperwithcode"></a> 
[3] Paper With Code, "Image Captioningd". Truy cập ở \href{https://paperswithcode.com/task/image-captioning}{https://paperswithcode.com/task/image-captioning} (Truy cập lần cuối vào 2022).

<a name="-asr_paperwithcode"> </a>
[4] Paper With Code, "Speech Recognitiond". Truy cập ở \href{https://paperswithcode.com/task/speech-recognition}{https://paperswithcode.com/task/speech-recognition} (Truy cập lần cuối vào 10/2022).

<a name="-ocr_paperwithcode"> </a>
[5] Paper With Code, "Optical Character Recognition'', Truy cập ở \href{https://paperswithcode.com/task/optical-character-recognition}{https://paperswithcode.com/task/optical-character-recognition} (Truy cập lần cuối vào 10/2022).

<a name="-batch_norm"> </a>
[6] S. Ioffe và C. Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shiftd". Truy cập ở \href{http://arxiv.org/abs/1502.03167}{http://arxiv.org/abs/1502.03167} (2015).

<a name="-im2latex_100k"> </a>
[7] K. Anssi, "im2latex-100kd". arXiv:1609.04938 [Data set]. Zenodo. \doi{10.5281/zenodo.56198} (2016)

<a name="-resnet"> </a>
[8] K. He, X. Zhang, S. Ren, J. Sun, "Deep Residual Learning for Image Recognitiond"., arXiv: 1512.03385, \doi{10.48550/ARXIV.1512.03385} (2015)

<a name="-resnet_gan"> </a>
[9] W. Meng, L. Huafeng, L. Fang, "Generative Adversarial Network based on Resnet for Conditional Image Restorationd". arXiv:1707.04881, \doi{10.48550/ARXIV.1707.04881} (2017)

<a name="-resnet_object_detection"> </a>
[10] H. Foysal, L. H. Youn, K. D. Seong, "Object Detection Based on VGG with ResNet Networkd".. pp. 1-3. \doi{10.23919/ELINFOCOM.2019.8706476} (2019).

<a name="-downsample"> </a>
[11] A. Youssef, "Image Downsampling and Upsampling Methodsd". Truy cập tại: \href{https://www2.seas.gwu.edu/~ayoussef/papers/ImageDownUpSampling-CISST99.pdf}{https://www2.seas.gwu.edu/~ayoussef/papers/ImageDownUpSampling-CISST99.pdf}

<a name="-luong_attention"> </a>
[12] M. T. Luong, H. Pham, C. D. Manning, "Effective Approaches to Attention-based Neural Machine Translationd". arXiv:1508.04025, \doi{10.48550/ARXIV.1508.04025} (2015).

<a name="-orthogonal"> </a>
[13] W. Hu, L. Xiao, J. Pennington, "Provable Benefit of Orthogonal Initialization in Optimizing Deep Linear Networksd". arXiv:2001.05992, \doi{10.48550/ARXIV.2001.05992} (2020).

<a name="-im2latex_100k_dataset"> </a>
[14] Kanervisto, Anssi. (2016). im2latex-100k , arXiv:1609.04938 [Data set]. Zenodo. \doi{10.5281/zenodo.56198}.

<a name="-adamw"> </a>
[15] I. Loshchilov, F. Hutter, "Decoupled Weight Decay Regularizationd". arXiv:1711.05101, \doi{10.48550/ARXIV.1711.05101} (2017).

<a name="-onecyclelr"> </a>
[16] L. N. Smith, N. Topin, "Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Ratesd". arXiv:1708.07120, \doi{10.48550/ARXIV.1708.07120} (2017).

<a name="-accumulate_gradient"> </a>
[17] J. Hermans, G. Spanakis, R. Möckel, "Accumulated Gradient Normalizationd". arXiv:1710.02368, \doi{10.48550/ARXIV.1710.02368} (2017).

<a name="-beam_search"> </a>
[18] Wikipedia, "Beam Searchd".. Truy cập tại: \href{https://en.wikipedia.org/wiki/Beam_search#:~:text=In%20computer%20science%2C%20beam%20search,that%20reduces%20its%20memory%20requirements.}{https://en.wikipedia.org/wiki/Beam_search#:~:text=In%20computer%20science%2C%20beam%20search,that%20reduces%20its%20memory%20requirements.}

<a name="-vit"> </a>
[19] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, N. Houlsby, "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scaled". arXiv:2010.11929, \doi{10.48550/ARXIV.2010.11929} (2020).

<a name="-im2latex_170k"> </a>
[20] Kaggle, "Image2Latex170kd".. Truy cập ở \href{https://www.kaggle.com/datasets/rvente/im2latex170k}{https://www.kaggle.com/datasets/rvente/im2latex170k}.