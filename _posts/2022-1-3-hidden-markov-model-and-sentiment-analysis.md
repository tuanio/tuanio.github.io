---
title: Mô hình Markov ẩn và bài toán phân tích cảm xúc văn bản
date: 2022-1-3 20:51:00 +/-0084
categories: [knowledge]
tags: [machine learning, probability, nlp]
toc: true
comments: true
published: true
math: true
---

### Nội dung
- [1. Định nghĩa](#-dinh-nghia)
- [2. Ba bài toán nền tảng](#-three-problems)
- [3. Bài toán phân tích cảm xúc văn bản](#-bai-toan-phan-tich-cam-xuc-van-ban)
    - [3.1 Giới thiệu bài toán phân tích cảm xúc văn bản](#-gioi-thieu-bai-toan)
    - [3.2 Bộ dữ liệu Financial News của Kaggle](#-bo-du-lieu)
    - [3.3 Mô hình bài toán](#-mo-hinh-bai-toan)
    - [3.4 Phương pháp thực hiện](#-phuong-phap)
- [4. Tổng kết](#-tong-ket)
- [5. Tham khảo](#-tham-khao)

<a name="-dinh-nghia"></a>
# 1. Định nghĩa

Ở bài viết về [Markov chain](/posts/markov-chain-va-bai-toan-sang-nay-an-gi/), chúng ta đã tìm hiểu về một mô hình được kết hợp bởi các trạng thái, các trạng thái cũng đồng thời cũng là kết quả của mô hình. Trong bài viết này, chúng ta sẽ tìm hiểu về mô hình Markov ẩn (Hidden Markov model - HMM), mà các trạng thái của mô hình lúc này sẽ không phải là thứ chúng ta có thể quan sát được.

Mô hình Markov ẩn là một mô hình thống kê được kết hợp bởi tập các trạng thái ẩn (hidden state) và tập các quan sát (observation). Mô hình Markov ẩn sử dụng tính chất Markov giống Markov chain, trạng thái hiện tại chỉ phụ thuộc vào trạng thái trước đó, ngoài ra các quan sát hiện tại chỉ phụ thuộc vào trạng thái hiện tại.

Mô hình Markov ẩn từng thống trị rất nhiều bài toán và lĩnh vực ở thập kỷ trước (chứng cứ là có rất nhiều bài báo được đăng tải liên quan đến mô hình Markov ẩn liên quan đến nhiều lĩnh vực tại thời điểm đó), nhất là trong lĩnh vực <a href="https://en.wikipedia.org/wiki/Speech_recognition" target="_blank">Nhận dạng giọng nói</a> (Speech Recognition). Trong lĩnh vực nhận dạng giọng nói, mô hình Markov ẩn đóng vai trò như một mô hình âm học đại diện cho một đơn vị nhận dạng giọng nói.

Mô hình Markov ẩn được kết hợp bởi 5 thành phần, ta có thể gọi một mô hình Markov ẩn là $\lambda =(Q, V, A, B, \pi)$ (có thể đơn giản hóa ký hiệu thành $\lambda =(A, B, \pi)$), trong đó:

- $Q=q_1, q_2, \cdots, q_N$ là tập gồm $N$ trạng thái ẩn, $X_t \in Q$ là giá trị ở thời điểm $t$ được lấy trong tập $Q$.
- $O=o_1, o_2, \cdots, o_T$ là một chuỗi gồm $T$ (là thời điểm cuối cùng) quan sát, mỗi quan sát được lấy từ tập giá trị duy nhất $V = \\{v_1, v_2, \cdots, v_V\\}$. 
- $A_{N\times N}$ là ma trận xác suất chuyển, được ký hiệu là $A=a_{ij}=\\{P(X_{t+1} = q_j|X_t = q_i)|1 \le i,j \le N\\}$.
 Ở đây, $a_{ij}$ đại diện cho xác suất chuyển từ trạng thái $i$ ở thời điểm $t$ sang trạng thái $j$ ở thời điểm $t+1$
- $B_{V\times N}$ là ma trận xác suất phát xạ (emission probability), và được ký hiệu bởi $B=b_i(k)=\\{P(O_t = v_k|X_t=Q_i)|1\le i\le N, 1 \le k \le V\\}$.
 $b_i(k)$ đại diện cho xác suất ký hiệu $v_k$ được phát xạ ra từ trạng thái $i$ tại thời điểm $t$.
- $\pi=\pi_i=\\{P(X_1=S_i)|1\le i \le n\\}$ 
là tập xác suất khởi tạo trạng thái.

Hình 2 mô tả trừu tượng cấu trúc của mô hình Markov ẩn được đề cập ở trên. Hình $(a)$ là sơ đồ gồm các tập trạng thái ẩn $q_i$ và các giá trị xác suất chuyển $a_{ij}$, trông giống hệt như một Markov chain. Hình $(b)$ mô tả: với mỗi trạng thái ẩn $q_i$, sẽ có một tập giá trị $v_k$ là tập giá trị sẽ được xuất ra với xác suất $b_i(k)$ tương ứng, và hình $(b)$ là điều khiến mô hình Markov ẩn khác với Markov chain.

<p>
    <img src="/assets/hmm/hmm_abstract.svg" alt="hmm_abstract"/>
    <em>Hình 1: Dạng và cấu tạo của mô hình Markov ẩn trừu tượng</em>
</p>

Hình dưới đây mô tả một dạng hiện thực của mô hình Markov ẩn, với $X_t \in Q$ và $O_t \in V$. Ta có một dạng của mô hình Markov ẩn theo thời gian thực. Dạng $\cdots$ (ba chấm) ở đây biểu thị các trạng thái trước đó và trạng thái tương lai cách thời điểm $t$ hơn $1$ đơn vị. Theo tính chất Markov, trạng thái hiện tại chỉ phụ thuộc vào trạng thái từ quá khứ cách nó một đơn vị, và trạng thái tương lai cũng chỉ phụ thuộc vào trạng thái hiện tại. Theo công thức toán học có thể mô tả là:
$$P(X_t|X_{t-1}, X_{t-2}, X_{t-3}, \cdots)=P(X_t|X_{t-1})$$.

<p>
    <img src="/assets/hmm/hmm_realisation.svg" alt="hmm_realisation" />
    <em>Hình 2: Một dạng hiện thực của mô hình Markov ẩn</em>
</p>

Do có cấu tạo như hình 2, mô hình Markov ẩn rất thích hợp trong những bài toán mô hình hóa chuỗi các giá trị. Trong thực tế, ta có thể xem các chuỗi giá trị là dữ liệu chúng ta có được từ thực tế và phân phối để lấy ra chuỗi giá trị kia ta không hề biết trước. Trong trường hợp này, ta có thể dùng mô hình Markov ẩn để mô hình hóa chuỗi giá trị đó để có được tập các trạng thái ẩn và phân phối xác suất thích hợp, cách để học và lấy ra các trạng thái ẩn sẽ được trình bày trong phần 2.

Phần tiếp theo, phần 2 sẽ giới thiệu ba bài toán nền tảng của mô hình Markov ẩn, tuy nền tảng nhưng là nền móng cho mọi bài toán phức tạp hơn trong thế giới thực chiến.

<a name="-three-problems"></a>
# 2. Ba bài toán nền tảng

Ở phần 1, tôi đã đi sơ lược về cấu tạo, cấu trúc và các thành phần đằng sau mô hình Markov ẩn. Đến thời điểm này, chắc hẳn bạn đọc sẽ thắc mắc cách sử dụng mô hình Markov ẩn như thế nào, vì thực tế, mô hình Markov ẩn có một cấu trúc dạng chuỗi tuần tự đặc biệt, trông rất khác so với các mô hình truyền thống như Linear Regression, Logistic Regression, Random Forest, ... .

Vì thế, mô hình Markov ẩn cũng sẽ có những cách sử dụng khác. Cụ thể hơn, để sử dụng mô hình Markov ẩn, ta bắt buộc phải giải quyết được 3 bài toán được mô tả dưới đây. Ba bài toán đó là:

- **Bài toán 1**: Đưa trước chuỗi quan sát $O=o_1, o_2, \cdots, o_T$ và mô hình $\lambda = (A, B, \pi)$. Làm cách nào để ta có thể tính hiệu quả
$P(O|\lambda)$, chính là xác suất để chuỗi quan sát xảy ra khi biết trước mô hình?
- **Bài toán 2**: Đưa trước chuỗi quan sát $O=o_1, o_2, \cdots, o_T$ và mô hình $\lambda = (A, B, \pi)$. Làm cách nào để ta có thể tìm được một chuỗi trạng thái ẩn $X=X_1, X_2, \cdots, X_T$ để giải thích tốt nhất cho chuỗi quan sát $O$?
- **Bài toán 3**: Làm cách nào để ta có thể điều chỉnh tham số của mô hình $\lambda = (A, B, \pi)$ để tối đa hóa xác suất 
$P(O|\lambda)$?

Bài toán 1 là bài toán đánh giá (evaluation problem), nghĩa là đi tính xác suất xảy ra của một chuỗi quan sát khi ta có được mô hình. Nếu nhìn ở một khía cạnh khác, đây chính là bài toán chấm điểm mô hình, nếu mô hình nào có xác suất 
$P(O|\lambda)$ cao hơn nghĩa là mô hình đó tốt hơn. Bài toán 2 là bài toán giải mã (decoding problem), có thể hiểu là ta đã có một chuỗi quan sát $O$ và ta có thể thấy, bây giờ ta phải tìm một chuỗi trạng thái ẩn tương ứng (có cùng kích cỡ) $X$ sao cho giải thích tốt nhất chuỗi quan sát $O$ kia. Bài toán 3 là bài toán học (learning problem), là bài toán quan trọng nhất. Vì nhờ bài toán 3, ta có thể tối ưu hóa các tham số của mô hình Markov ẩn $\lambda$ đến mức hội tụ, sử dụng cho nhiều bài toán thực tế khác nhau.

Cả ba bài toán trên đều có cách giải rất đơn giản, đó là thế vào và thử, tuy nhiên độ phức tạp tính toán sẽ rất cao, nên người ta dùng kỹ thuật quy hoạch động (<a href="https://en.wikipedia.org/wiki/Dynamic_programming" target="_blank">dynamic programming</a>) để tối ưu, giúp giải quyết cả 3 vấn đề một cách quy nạp và theo độ phức tạp đa thức. Cụ thể bài toán 1 có thể giải với thuật toán <a href="https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm" target="_blank">forward-backward</a>, bài toán 2 sẽ giải bằng thuật toán <a href="https://en.wikipedia.org/wiki/Viterbi_algorithm" target="_blank">Viterbi</a> và bài toán 3 sẽ giải bằng thuật toán <a href="https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm" target="_blank">Baum-Welch</a>.

Trong phần này, tôi chỉ đi giới thiệu về ba bài toán, về cách giải sẽ không được đề cập đến, bạn đọc có hứng thú với lời giải cho ba bài toán có thể tham khảo [[3]](#-reference-3), đây là bài báo rất chất lượng về mô hình Markov ẩn, là nền tảng cho bất cứ ai mới bắt đầu tìm hiểu về mô hình Markov ẩn. Nếu gặp khó khăn trong việc hiện thực thuật toán, bạn đọc có thể tham khảo đến <a href="https://github.com/tuanio/hmm" target="_blank">github</a> của tôi, tôi cũng đã đọc bài báo số [[3]](#-reference-3) và hiện thực thành công.

<a name="-bai-toan-phan-tich-cam-xuc-van-ban"></a>
# 3. Bài toán phân tích cảm xúc văn bản

<a name="-gioi-thieu-bai-toan"></a>
## 3.1 Giới thiệu bài toán phân tích cảm xúc văn bản
Phân tích cảm xúc văn bản (<a href="https://en.wikipedia.org/wiki/Sentiment_analysis" target="_blank">sentiment analysis</a>) là bài toán được nghiên cứu trong lĩnh vực Xử lý ngôn ngữ tự nhiên. Mục tiêu của bài toán là tìm ra cảm xúc (*tích cực*, *tiêu cực*, *trung tính*) của một câu chữ trong một lĩnh vực cụ thể nào đó. Bài toán này rất được ưa chuộng trong các công ty mà lượng dữ liệu về chữ của họ lớn, họ có thể khai thác thông tin từ nguồn dữ liệu của họ, từ đó hiểu được khách hàng của họ cần gì. Ví dụ như các bình luận trên shopee hay tiki là một ví dụ, một câu "Tôi rất thích sản phẩm này" sẽ được đánh nhãn là *tích cực*, câu "Sản phẩm này nhăn nheo quá" sẽ được gán nhãn là *tiêu cực*, một trường hợp khác có nhãn là *trung tính*, không rõ ràng *tích cực* hay *tiêu cực*, ví dụ như câu "Hôm nay tôi vừa nhận được sản phẩm này".

Hiện tại, bài toán này có thể giải quyết bằng những phương pháp Machine Learning hoặc mạnh hơn là Deep Learning, chi tiết bạn đọc có thể tìm hiểu ở <a href="https://paperswithcode.com/task/sentiment-analysis" target="_blank">đây</a>. Nhưng trong phạm vi bài viết này, chúng ta sẽ tiếp cận với một hướng khác, đó là giải quyết bài toán này bằng mô hình Markov ẩn.

<a name="-bo-du-lieu"></a>
## 3.2 Bộ dữ liệu Financial News của Kaggle

Bộ dữ liệu chúng ta sẽ đi nghiên cứu là bộ <a href="https://www.kaggle.com/ankurzing/sentiment-analysis-for-financial-news" target="_blank">Financial News</a> được lấy trên Kaggle. Dữ liệu gồm 2 cột, 4837 hàng, cột thứ nhất là nhãn, tức là cảm xúc của văn bản đã được gắn từ trước, gồm 3 giá trị: `positive`, `neutral`, `negative`. Cột thứ hai là văn bản. Dữ liệu này đầy đủ và đơn giản để sử dụng trong bài toán này.

<a name="-mo-hinh-bai-toan"></a>
## 3.3 Mô hình bài toán

Khi ứng dụng mô hình Markov ẩn vào bài toán phân lớp (classification) như chúng ta đang định làm, chúng ta phải mô hình hóa một số lượng mô hình Markov ẩn riêng biệt bằng với số lượng lớp của bài toán. Nếu lấy bộ dữ liệu Financial News kia làm chuẩn, ta sẽ có 3 mô hình Markov ẩn tương ứng với 3 lớp `positive`, `neutral` và `negative`.

Mô hình Markov ẩn sẽ làm tốt công việc của mình trong việc mô hình hóa phân phối xác suất của riêng từng lớp. Nếu coi tập dữ liệu là $O$ và có tổng cộng 3 mô hình Markov ẩn tương ứng với 3 lớp thì lớp dự đoán khi ta đưa dữ liệu mới vào sẽ theo công thức dưới đây:

$$C^\star = \underset{C}{\mathrm{argmax }} P(O|\lambda_C)$$

Trong đó:
- $C$ là lớp (nhãn) và $C^\star$ là lớp dự đoán.
- $\lambda_C$ là mô hình Markov ẩn tương ứng với mỗi lớp.
- $P(O|\lambda_C)$
chính là bài toán 1, bài toán đánh giá mô hình.

Bất cứ mô hình Machine Learning nào cũng sẽ có giai đoạn huấn luyện, mô hình Markov ẩn cũng không ngoại lệ. Hình (3a) mô tả quy trình này, ban đầu ta có một tập dữ liệu $O$ (có thể là nhiều $O$) và $n$ lớp (nhãn) tương ứng. Ta sẽ chia tập dữ liệu ra thành $n$ tập dữ liệu nhỏ hơn tương ứng với $n$ nhãn. Sau đó dùng thuật toán Baum-Welch (bài toán số 3) để huấn luyện cho mô hình $\lambda_{C_i}$ tương ứng. Kết thúc quá trình huấn luyện, ta được $n$ mô hình Markov ẩn tương ứng với $n$ nhãn lớp.

Để có thể sử dụng $n$ mô hình kia trong quá trình kiểm thử hoặc đi dự đoán. Ta cần đưa dữ liệu kiểm thử cho cả $n$ mô hình Markov ẩn, sau đó đi tìm các xác suất 
$P(O|\lambda_{C_i})$ (bài toán số 1) và chọn nhãn $C$ có giá trị xác suất lớn nhất, nhãn $C$ này sẽ là nhãn dự đoán cho chuỗi quan sát $O$ ta đưa vào. Hình (3b) mô tả rõ quy trình này.

<p>
    <img src="/assets/hmm/hmm_diagram.svg" alt="hmm_diagram" />
    <em>Hình 3: Sơ đồ của mô hình Markov ẩn (a) trong quá trình huấn luyện và (b) trong quá trình kiểm thử</em>
</p>

<a name="-phuong-phap"></a>
## 3.4 Phương pháp thực hiện

Bây giờ chúng ta sẽ đi đến phần hiện thực bài toán, tôi sẽ sử dụng ngôn ngữ lập trình Python với các thư viện ở ô code dưới đây.

⚠️ **Lưu ý**: khi code báo lỗi thư viện, các bạn có thể tự cài thư viện thông qua `pip install {tên thư viện}`.

````python
import numpy as np # thư viện tính toán 
import pandas as pd # đọc file csv
import concurrent.futures as cf # thư viện giúp code python chạy đa luồng
from hmmlearn import hmm # thư viện mô hình Markov ẩn 
from sklearn.cluster import KMeans # lượng hóa vector
from sklearn.metrics import accuracy_score # đo độ chính xác của mô hình
from sklearn.decomposition import TruncatedSVD # giảm chiều dữ liệu
from sklearn.model_selection import train_test_split # chia tập dữ liệu train|test
from sklearn.feature_extraction.text import TfidfVectorizer # tạo feature cho mô hình từ chữ
````

👉 Chúng ta sẽ đi qua các bước như sau:
1. Tạo feature dữ liệu số từ dữ liệu chữ có sẵn bằng TF-IDF.
2. Lượng hóa vector (vector quantization) dữ liệu số liên tục thành dạng định tính có thể đem đi huấn luyện.
3. Chia tập dữ liệu huấn luyện, kiểm thử tương ứng.
4. Huấn luyện bộ mô hình Markov ẩn với tập dữ liệu huấn luyện.
5. Đánh giá mô hình Markov ẩn thông qua tập dữ liệu kiểm thử.

Trước tiên, ta sẽ đọc dữ liệu để có thể chuẩn bị cho bước tạo feature cho mô hình. Dữ liệu đã được đề cập trong phần 3.2.

````python
df = pd.read_csv('all-data.csv', encoding="ISO-8859-1", header=None, names=['label', 'text'])
````

Để trạng thái của code không thay đổi qua mỗi lần chạy, ta nên gán cụ thể giá trị `random state` cho các thư viện. Dưới đây tôi định nghĩa biến `rs` là giá trị `random state` để dùng cho các code sau. Giá trị các bạn có thể thay đổi bất kỳ.

````python
rs = 8
````

Tạo biến `corpus` để gán dữ liệu chữ vào, tiện sử dụng về sau.

````python
corpus = df['text'].values
````

Như các mô hình Machine Learning truyền thống khác, mô hình Markov ẩn sẽ chỉ làm việc được với các giá trị số. Mà dữ liệu ban đầu của chúng ta là dữ liệu dạng chữ, nên ta phải chuyển từ chữ sang số. Để làm như vậy, ta sử dụng TF-IDF để tính toán các giá trị trọng số để đại diện cho từng từ một trong bộ ngữ liệu ban đầu. Chi tiết hơn về TF-IDF, bạn đọc có thể tham khảo ở <a href="https://en.wikipedia.org/wiki/Tf%E2%80%93idf" target="_blank">đây</a>. Còn trong Python, ta sẽ tính bằng đoạn code sau:

````python
tfidf = TfidfVectorizer(stop_words='english')
transformed = tfidf.fit_transform(corpus)

print("Kích cỡ dữ liệu:", transformed.shape)
````
````plain
Kích cỡ dữ liệu: (4846, 9820)
````

Như bạn đọc cũng đã thấy, có tận 9820 cột dữ liệu được tạo ra, như vậy là quá nhiều, ta phải dùng cách nào đó để giữ lại các thông tin quan trọng nhất, giảm bớt số lượng cột lại, nhờ đó giúp giảm thời gian huấn luyện và kiểm thử, mô hình cũng không phải học những thông tin dư thừa. Trong trường hợp này, ta sẽ dùng `Truncated SVD` với số lượng cột ta muốn giữ lại là 300. Chi tiết về `Truncated SVD`, bạn đọc có thể tham khảo ở <a href="https://machinelearningcoban.com/2017/06/07/svd/#-truncated-svd" target="_blank">blog machine learning cơ bản</a>.

````python
svd = TruncatedSVD(n_components=300, random_state=rs)
X_transformed = svd.fit_transform(transformed)

print("Kích cỡ dữ liệu:", X_transformed.shape)
print(X_transformed)
````
````plain
Kích cỡ dữ liệu: (4846, 300)
[[ 2.74853772e-02  1.23546023e-01 -9.18267054e-02 ... -1.04585446e-02
   6.31456379e-02  1.64352977e-02]
 [ 1.71779475e-02  6.60849287e-02 -3.16730100e-02 ...  3.16069539e-02
   2.32036018e-02 -8.03645790e-03]
 [ 2.53099565e-02  8.71819162e-02 -5.00550792e-02 ... -5.08493999e-02
  -6.52592321e-02 -4.16690704e-02]
 ...
 [ 6.29146372e-01 -1.92710754e-01  3.24730118e-02 ...  3.28086253e-02
  -1.07719957e-02 -8.53337727e-04]
 [ 6.66497950e-01 -1.41546038e-01  1.81541966e-03 ...  7.96125651e-03
  -2.97791037e-03 -1.60182069e-04]
 [ 9.54134895e-02  1.71111951e-01 -6.08651229e-02 ... -1.30032385e-02
   3.79181183e-02  1.24508153e-02]]
````

Ngoài vấn đề có quá nhiều cột trong feature đã được giải quyết, ta còn gặp thêm một vấn đề nữa đó là dữ liệu không phù hợp với mô hình Markov ẩn. Như đã tìm hiểu trên phần 1, các quan sát $O$ của mô hình Markov ẩn được lấy từ một tập $V$ phần tử, vì thế, dữ liệu đưa vào cho mô hình Markov ẩn phải là dạng định tính.

Để giải quyết vấn đề trên, ta có thể dùng một kỹ thuật được gọi là lượng hóa vector (<a href="https://en.wikipedia.org/wiki/Vector_quantization" target="_blank">vector quantization</a>). Lượng hóa vector có thể hiểu đơn giản là phân cụm các giá trị liên tục thành một tập các cụm có sự giống nhau. Chỉ số của các cụm bây giờ có thể coi như là các giá trị được lấy trong tập $V = \text{số cụm}$ phần tử. Giá trị trong cùng một tập sẽ có cùng một chỉ số này. Ở phần hiện thực, tôi sẽ đi lượng hóa vector bằng thuật toán <a href="https://en.wikipedia.org/wiki/K-means_clustering" target="_blank">K-Means</a> với số cụm là 30.

````python
X_cluster = X_transformed.reshape(-1, 1) 

vq = KMeans(n_clusters=30) # vector quantization
vq.fit(X_cluster)

def map_vq(x):
    return vq.predict(x.reshape(-1, 1))

with cf.ThreadPoolExecutor() as exe:
    X = np.array(list(exe.map(map_vq, X_transformed)))

print(X)
````
````plain
[[ 9 11  0 ...  8 16 22]
 [27 16 20 ...  9 27  8]
 [ 9  3 19 ... 19 28 10]
 ...
 [26 24  4 ...  4  8 13]
 [26 24 13 ...  1 29 13]
 [ 3 18 28 ... 17  4 22]]
````

Như bạn có thể thấy, dữ liệu định lượng được lấy ra từ TF-IDF đã chuyển thành dạng số nguyên, là chỉ số của các cụm. Bây giờ, ta sẽ đi phân chia dữ liệu thành hai tập: huấn luyện và kiểm thử với tỉ lệ 8:2.

````python
y = df['label']

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=rs)
````

Sau khi đã có dữ liệu, ta sẽ đi tạo mô hình Markov ẩn cho bài toán này rồi mới huấn luyện. Ta sẽ tạo cấu trúc mô hình giống như hình 3.

````python
class HMMSystem:
    def __init__(self, n_components=10, random_state=rs):
        self.n_components = n_components
        self.random_state = random_state
    
    def fit(self, X, y):
        self.labels = np.unique(y)
        self.X = X
        self.y = y
        self.hmm_models = {}
        for c in self.labels:
            with cf.ThreadPoolExecutor() as exe:
                self.hmm_models = list(exe.map(self._create_model, self.labels))
            self.hmm_models = dict(zip(self.labels, self.hmm_models))
        return self
        
    def predict(self, X):
        with cf.ThreadPoolExecutor() as exe:
            pred = np.array(list(exe.map(self._find_class, X)))
        return pred
    
    def _create_model(self, label):
        model = hmm.MultinomialHMM(
            n_components=self.n_components,
            random_state=self.random_state
        ).fit(self.X[self.y == label])
        return model
    
    def _find_class(self, data):
        def _decode(model):
            return model.decode([data])[0]

        with cf.ThreadPoolExecutor() as exe:
            logprobs = list(exe.map(_decode, self.hmm_models.values()))

        return self.labels[np.argmax(logprobs)]
````

Huấn luyện mô hình với tập dữ liệu huấn luyện. Ta sẽ chọn số lượng trạng thái ẩn $Q$ là 8.

⚠️ **Lưu ý:** quá trình huấn luyện có thể hơi lâu, khoảng vài phút.

````python
model = HMMSystem(8, rs)
model.fit(X, y)
````

Dùng tập kiểm thử để dự đoán với mô hình đã huấn luyện.

````python
ytest_pred = model.predict(Xtest)
ytrain_pred = model.predict(Xtrain)
````

Cuối cùng là đi đánh giá trên cả hai tập dữ liệu đã dự đoán.

````python
acc_train = accuracy_score(ytrain, ytrain_pred)
acc_test = accuracy_score(ytest, ytest_pred)

print("Độ chính xác tập huấn luyện: %.2f/1" % acc_train)
print("Độ chính xác tập kiểm thử: %.2f/1" % acc_test)
````
````plain
Độ chính xác tập huấn luyện: 0.46/1
Độ chính xác tập kiểm thử: 0.48/1
````

Độ chính xác của mô hình là $48\%$ trên tập kiểm thử, mức độ chính xác này vào khoảng giữa, 50:50, nên kết quả dự đoán của mô hình còn khá rủi ro.

Toàn bộ code Python hiện thực sẽ để ở <a href="https://github.com/tuanio/sentiment-analysis-discrete-hmm" target="_blank">đây</a>.

<a name="-tong-ket"></a>
# 4. Tổng kết

Trong bài viết này, chúng ta đã đi qua sơ lược về định nghĩa, các thành phần và các bài toán của mô hình Markov ẩn. Từ đó ứng dụng mô hình Markov ẩn vào bài toán phân tích cảm xúc văn bản. Độ chính xác của mô hình không quá cao, nhưng nó giúp bạn đọc có thể hiểu thêm về mô hình này. Bạn đọc có thể đọc thêm về mô hình Markov ẩn ở các bài báo và địa chỉ website trong phần 5.

<a name="-tham-khao"></a>
# 5. Tham khảo


[1] https://web.stanford.edu/~jurafsky/slp3/A.pdf

[2] https://en.wikipedia.org/wiki/Hidden_Markov_model

[3] A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition. Rabiner. 1989.
<a name="-reference-3" ></a>

[4] A Systematic Review of Hidden Markov Models and Their Applications. Bhavya Mor, Sunita Garhwal & Ajay Kumar. 2021.

[5] Hidden Markov Models for Sentiment Analysis in Social Medias. Isidoros Perikos. 2019.