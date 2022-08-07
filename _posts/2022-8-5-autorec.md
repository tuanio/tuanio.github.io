---
title: "AutoRec: Autoencoder dành cho Collaborative Filtering"
date: 2022-8-5 01:07:00 +/-0084
categories: [knowledge]
tags: [machine learning, collaborative filtering, recommendation system, movielens dataset, autoencoder, supervised learning, auto-associative neural network]
toc: true
comments: true
published: false
math: true
---

### Nội dung
1. Hệ thống gợi ý và phương pháp Lọc cộng tác
2. Kiến trúc AutoRec
3. Thực nghiệm với bộ dữ liệu MovieLens

# 1. Hệ thống gợi ý và phương pháp Lọc cộng tác

Hệ thống gợi ý (recommendation system), là hệ thống giúp đưa ra gợi ý những sản phẩm thích hợp nhất đối với những người dùng cụ thể nào đó. Thường quá trình gợi ý này phụ thuộc vào nhiều yếu tố, ví dụ như sản phẩm nào họ đã từng mua, những sản phẩm nào họ đã tương tác (nút like), nhạc nào họ đã từng nghe, món nào họ đã từng ăn hoặc dựa trên những người họ quen biết trên nền tảng điện tử đó, hoặc dựa trên những người dùng có hành vi tương đối giống họ.

Vâng, là dựa vào những người dùng có hành vi tương đối giống họ, đây là giải thích ngắn gọn cho phương pháp lọc cộng tác. Phương pháp lọc cộng tác (collaborative filtering) dùng dữ liệu từ những người dùng có hành vi tương đối giống họ (đánh giá dựa trên khoảng cách được tính từ một số yếu tố như có phải bạn bè hay không, các bộ phim đã yêu thích, thể loại đã xem, yêu thích, mức đánh giá ...) để đưa ra gợi ý cho người dùng đó tương tự như những người dùng này.

Có thể bắt gặp nhiều phương pháp lọc cộng tác khác nhau khi tìm từ khoá ***collaborative filtering***, ví dụ như Neighborhood-based, Matrix Factorization (MF), Restricted Bolzmann Machine-based (RBM-based), ..., vì các phương pháp này đã được thử nghiệm từ lâu. Nhưng phương pháp mà tôi trình bày trong bài viết này có lẽ vẫn còn mới hơn các phương pháp kể trên nên có thể tìm kiếm sẽ không thấy được, paper gốc của phương pháp này được công bố vào 5/2015.

# 2. Kiến trúc AutoRec

Trong paper gốc của AutoRec, tác giả sử dụng và chỉnh sửa lại dạng mạng nơ-ron liên kết tự động (auto-associative neural network) Autoencoder. Tác giả sử dụng Autoencoder vì sự thành công của mạng nơ-ron sâu (deep neural network) vào khoảng thời gian tác giả nghiên cứu kiến trúc này. Tác giả tin rằng, AutoRec sẽ có điểm lợi hơn các phuơng pháp Matrix Factorization và RBM-based về thời gian tính toán và biểu diễn dữ liệu.

Ta cùng quay lại ngữ cảnh của bài toán lọc cộng tác (CF). Đối với một hệ thống điện tử, ta sẽ có $m$ user và $n$ item và trong ngữ cảnh CF, ta sẽ có thêm một ma trận đánh giá (user $u$ đánh giá item $i$) quan sát một phần (không đầy, sẽ có những chỗ là giá trị $0$) user-item $R \in \mathbb{R}^{m \times n}$. Trong rating-based, mỗi user $u \in U = \\{1 \cdots m\\}$ được biểu diễn bởi một vector rating quan sát một phần $\mathrm{r}^{(u)} = (R_{u1}, \cdots, R_{un}) \in \mathbb{R}^n$. Còn trong user-based, mỗi item $i \in I = \\{1 \cdots n\\}$ được biểu diễn bởi một vector rating quan sát một phần $\mathrm{r}^{(i)} = (R_{1i}, \cdots, R_{mi}) \in \mathbb{R}^m$. Mục tiêu của việc sử dụng Autoencoder trong rating-based (hoặc item-based) là nhận một vector quan sát một phần $\mathrm{r}^{(i)}$ (hoặc $\mathrm{r}^{(u)}$), chiếu nó xuống không gian ẩn có có số chiều thấp hơn, và tái tạo lại $\mathrm{r}^{(i)}$ (hoặc $\mathrm{r}^{(u)}$) tương ứng để dự đoán những rating bị thiếu (những vị trí mà giá trị hiện tại là $0$ hoặc rỗng) cho mục đích gợi ý (từ rating dự đoán ta có thể sắp xếp lại các sản phẩm phù hợp để đưa cho người dùng).

Cụ thể về mặt toán học, ta có một tập $\mathrm{S} \in \mathbb{R}^d$ gồm các vector và một giá trị $k \in \mathbb{N}_{+}$, Autoencoder sẽ giải:

$$\underset{\theta}{\mathrm{min}} \sum_{\mathrm{r} \in \mathrm{S}}||\mathrm{r} - h(\mathrm{r}; \theta)||^2_{2}$$

trong đó $h(\mathrm{r}; \theta)$ là giá trị rating được tái tạo của $\mathrm{r} \in \mathbb{R}^d$,

$$h(\mathrm{r}; \theta) = f(\mathrm{W} \cdot g(\mathrm{V}\mathrm{r}+\mathrm{\mu}) + \mathrm{b})$$

với các activation function $f(\cdot)$, $g(\cdot)$. Và bộ các tham số của mô hình $\theta = \\{\mathrm{W}, \mathrm{V}, \mathrm{r}, \mathrm{b}\\}$, có kích cỡ lần lượt là $\mathrm{W} \in \mathbb{R}^{d\times k}$, $\mathrm{V} \in \mathbb{R}^{k\times d}$ và các bias $\mathrm{\mu} \in \mathbb{R}^k$, $\mathrm{b} \in \mathbb{R}^d$. Hàm mục tiêu của Autoencoder ở trên là hàm mục tiêu điển hình của mạng nơ-ron tự liên kết, một lớp ẩn có $k$ chiều, và bộ tham số $\theta$ sẽ được học thông qua back-propagation.

Với AutoRec, tác giả sử dụng lại công thức của hàm mục tiêu Autoencoder phía trên, với 2 thay đổi:
1. Chỉ cập nhật những trọng số tương ứng với những quan sát đã có, bằng cách nhân với mask trong quá trình huấn luyện, ta sẽ loại bỏ cập nhật được những quan sát chưa có.
2. Chỉnh hoá các tham số của mô hình để tránh việc Overfit xảy ra.

Vì thế, khi áp dụng hàm mục tiêu của Autoencoder với 2 thay đổi trên vào bộ vector rating của Item-based (gọi là I-AutoRec) $\\{\mathrm{r}^{(i)}\\}^n_{i=1}$ và tham số chỉnh hoá $\lambda > 0$ bất kỳ, ta sẽ có hàm mục tiêu của AutoRec:

$$\underset{\theta}{\mathrm{min}} \sum_{i=1}^n||\mathrm{r^{(i)}} - h(\mathrm{r^{(i)}}; \theta)||^2_{\mathcal{O}} + \dfrac{\lambda}{2}\cdot (||\mathrm{W}||^{2}_{F} + ||\mathrm{V}||^{2}_{F})$$

trong đó, kí hiệu $$ \|\cdot\|^{2}_{\mathcal{O}} $$ thể hiện rằng chỉ xem xét những giá trị đã quan sát được (đã rating). Với User-based (gọi là U-AutoRec), ta áp dụng tương tự đối với tập vector rating $\\{\mathrm{r}^{(u)}\\}^m_{u=1}$. Tổng quan lại, I-AutoRec sẽ yêu cầu ước lượng $2mk + m + k$ tham số tất cả. Khi đã học được tham số $\hat{\theta}$, dự đoán rating của user $u$ dành cho item $i$ là:

$$\mathrm{R}^{ui} = (h(\mathrm{r}^{(i))}; \hat{\theta}))_{u}$$

<p>
    <img src="/assets/autorec/autorec.jpeg" alt="autorec"/>
    <em>Hình 1. Cấu trúc của Item-based AutoRec, các node màu xám thể hiện rating đã có, màu trắng thể hiện không có rating và là giá trị cần dự đoán.</em>
</p>

Ở trong nghiên cứu, tác giả đề cập đến việc sử dụng các loại activation function khác nhau cho $f(\cdot)$ và $g(\cdot)$, bảng dưới đây đánh giá RMSE của các kết hợp (càng thấp càng tốt). Trong đó Identity là không có hàm kích hoạt, còn Sigmoid được định nghĩa ở <a href="https://en.wikipedia.org/wiki/Sigmoid_function" target="_blank">đây</a>.

|$f(\cdot)$|$g(\cdot)$|RMSE|
|---|---|---|
|Identity|Identity|$0.872$|
|Sigmoid|Identity|$0.852$|
|Identity|Sigmoid|$\textbf{0.831}$|
|Sigmoid|Sigmoid|$0.836$|

Trong paper gốc, tác giả đề cập rằng AutoRec rất khác so với các mô hình dành cho CF lúc đó. Cụ thể, khi so sánh AutoRec với RBM-CF, ta có:
1. RBM-CF là dạng mô hình tổng hợp xác suất (generative, probabilistic model) dựa trên RBM. Còn AutoRec là mô hình phân biệt (discriminative model) dựa trên Autoencoder.
2. RBM-CF ước lượng các tham số bằng tối đa hoá log khả năng (maximizing log likelihood), còn AutoRec trực tiếp minimize RMSE, mà đây cũng là cách đánh giá hiệu suất kinh điển trong bài toán dự đoán rating.
3. RBM-CF huấn luyện bằng contrastive divergence, còn AutoRec sử dụng gradient-based backpropagation, mà nhanh hơn nhiều so với RBM-CF.
4. RBM-CF chỉ sử dụng được cho rating dạng rời rạc. Còn AutoRec dùng cho rating dạng liên tục. Với $r$ rating, RBM-CF phải tốn $nkr$ hoặc $mkr$ tham số, trong khi đó AutoRec không quan tâm đến số lượng $r$ nên dùng ít bộ nhớ hơn và khó overfit hơn.

Còn khi AutoRec so sánh với Matrix Factorization thì:
1. MF nhúng cả item và user vào không gian ẩn, còn I-AutoRec chỉ nhúng item (U-AutoRec chỉ nhúng user), nên mô hình sẽ nhẹ hơn.
2. MF học một cách biểu diễn ẩn tuyến tính (linear latent representation), còn AutoRec có thể biểu diễn dữ liệu ẩn theo dạng phi tuyến (nonlinear latent representation) thông qua hàm kích hoạt $g(\cdot)$, mà sẽ tạo được sự tổng quát hoá dữ liệu tốt hơn nhiều.

Trong phần sau, chúng ta sẽ đi thực nghiệm AutoRec bằng Pytorch trên bộ dữ liệu Movielens.

# 3. Thực nghiệm với bộ dữ liệu Movielens