# Feature Specification: Hoàn thiện Phase 1 — độ chính xác, hướng mắt, nói chuyện và chuẩn hóa cấu trúc

**Feature Branch**: `002-phase1-hardening`

**Created**: 2026-10-16

**Status**: Approved (người dùng duyệt phạm vi trong hội thoại ngày 2026-10-16)

**Input**: User description: "Refactor dự án về đúng convention và pattern; phase 1 hiện chỉ phát hiện cử
chỉ rất cơ bản và chưa chính xác — làm phase 1 hoàn chỉnh nhất có thể. Tự tìm video mẫu; cho chọn hai
kiểu bài thi (thi giấy, thi máy); implement luôn nói chuyện và hướng mắt trong phase 1."

**Spec gốc**: [001-smart-exam-proctoring](../001-smart-exam-proctoring/spec.md). Spec này kéo các đề
xuất FR-12..FR-19 của spec 001 vào phạm vi triển khai và bổ sung yêu cầu mới.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Đo chất lượng phát hiện bằng video có nhãn (Priority: P1)

Người phát triển hoặc người vận hành chạy ứng dụng ở chế độ phát lại trên một bộ video có nhãn khoảng
thời gian hành vi và nhận báo cáo cho từng loại sự kiện: số sự kiện đúng, bỏ sót, báo nhầm mỗi phút,
độ trễ từ lúc hành vi bắt đầu tới lúc cảnh báo, và tốc độ xử lý. Báo cáo được lưu để so sánh giữa các
lần thay đổi.

**Why this priority**: Không có số đo thì không thể nói phát hiện "chính xác hơn". Hiến pháp (nguyên
tắc VI) yêu cầu đánh giá bằng video đại diện trước khi coi một tính năng AI là hoàn thành. Mọi story
còn lại dựa vào story này để chứng minh kết quả.

**Independent Test**: Chạy phát lại trên một video có file nhãn, nhận báo cáo có đủ chỉ số cho từng
loại sự kiện; chạy lại cùng video và cùng cấu hình cho kết quả giống hệt.

**Acceptance Scenarios**:

1. **Given** một video và file nhãn khoảng thời gian, **When** chạy phát lại, **Then** báo cáo liệt kê
   từng loại sự kiện với số đúng, số bỏ sót, số báo nhầm, báo nhầm mỗi phút và độ trễ trung vị.
2. **Given** hai lần chạy với cùng video và cấu hình, **When** so sánh kết quả, **Then** danh sách sự
   kiện sinh ra giống hệt nhau.
3. **Given** báo cáo của lần trước được lưu làm mốc, **When** chạy lần mới, **Then** báo cáo chỉ rõ
   chỉ số nào tốt lên hoặc xấu đi so với mốc.
4. **Given** một video không đọc được hoặc file nhãn sai định dạng, **When** chạy phát lại, **Then**
   công cụ dừng với thông báo lỗi chỉ rõ tệp và dòng lỗi.

---

### User Story 2 - Chọn kiểu bài thi: thi giấy hoặc thi máy (Priority: P1)

Trước khi bắt đầu phiên, giám thị chọn kiểu bài thi. Với thi giấy, cúi nhìn xuống bàn là hành vi bình
thường và không tạo cảnh báo; liếc ngang, quay đầu, điện thoại, người thứ hai vẫn được cảnh báo. Với
thi máy, nhìn ra ngoài vùng màn hình hoặc cúi đầu kéo dài mới là tín hiệu cần chú ý, còn liếc nhanh
xuống bàn phím thì không.

**Why this priority**: Cùng một cử chỉ có ý nghĩa khác nhau tùy kiểu bài thi. Áp một bộ luật cho cả hai
gây báo nhầm hàng loạt (thi giấy) hoặc bỏ sót (thi máy).

**Independent Test**: Phát lại cùng một đoạn video cúi đầu nhìn bàn ở hai kiểu bài thi; thi giấy không
có cảnh báo cúi đầu, thi máy có cảnh báo khi cúi đủ lâu.

**Acceptance Scenarios**:

1. **Given** màn hình bắt đầu phiên, **When** giám thị mở danh sách kiểu bài thi, **Then** có hai lựa
   chọn "Thi giấy" và "Thi máy", mặc định là lựa chọn của lần gần nhất (lần đầu là "Thi máy").
2. **Given** phiên thi giấy, **When** thí sinh cúi nhìn xuống bàn 30 giây liên tục, **Then** không có
   cảnh báo cúi đầu.
3. **Given** phiên thi máy, **When** thí sinh cúi đầu quá thời lượng cấu hình, **Then** có một cảnh báo
   cúi đầu; liếc xuống bàn phím ngắn hơn thời lượng đó không tạo cảnh báo.
4. **Given** một phiên đang chạy, **When** xem tệp sự kiện, **Then** kiểu bài thi được ghi trong ảnh
   chụp chính sách của phiên và không đổi được giữa phiên.

---

### User Story 3 - Tín hiệu chính chính xác và không trùng lặp (Priority: P1)

Giám thị nhận cảnh báo đáng tin cho các hành vi chính: dùng điện thoại, có sách, quay đầu trái/phải,
cúi đầu, quay đầu lặp lại, vắng người, có thêm người, rời chỗ, che camera, không thấy mặt. Mỗi lần
hành vi xảy ra chỉ tạo một cảnh báo; các cảnh báo cùng gốc (ví dụ thí sinh rời đi làm mất cả người lẫn
mặt) không bị cộng điểm trùng.

**Why this priority**: Đây là giá trị cốt lõi của phase 1. Bản hiện tại có lỗi nối cấu hình, ghép sai
dữ liệu giữa các frame, và nhiều cảnh báo chồng nhau cho cùng một sự việc.

**Independent Test**: Chạy bộ video đánh giá (US1) trước và sau thay đổi; báo nhầm mỗi phút trên đoạn
ngồi bình thường giảm, độ phủ của các hành vi chính không giảm.

**Acceptance Scenarios**:

1. **Given** thí sinh cầm điện thoại liên tục 20 giây và điện thoại bị che khuất ngắn dưới 1 giây vài
   lần, **When** phát hiện, **Then** chỉ có một cảnh báo điện thoại cho cả đoạn.
2. **Given** thí sinh rời khỏi khung hình 10 giây, **When** phát hiện, **Then** cảnh báo vắng người là
   cảnh báo chính; "không thấy mặt" và "rời chỗ" được gắn làm tín hiệu phụ và không cộng thêm điểm.
3. **Given** cấu hình thời lượng riêng cho "có thêm người", "rời chỗ", "che camera", **When** chạy,
   **Then** mỗi luật dùng đúng thời lượng của nó.
4. **Given** thí sinh cầm bộ bài, ví hoặc điều khiển, **When** phát hiện, **Then** tỉ lệ cảnh báo điện
   thoại nhầm được đo riêng trong báo cáo đánh giá.
5. **Given** đầu thí sinh lệch sẵn khi bắt đầu phiên, **When** ứng dụng đề xuất hiệu chỉnh tự động,
   **Then** mốc chỉ được nhận khi đầu đứng yên và nhìn về phía màn hình; mốc không chắc chắn bị từ chối
   và giám thị vẫn hiệu chỉnh thủ công được.

---

### User Story 4 - Phát hiện hướng mắt (Priority: P2)

Sau bước hiệu chỉnh ngắn (nhìn vào các điểm trên màn hình), ứng dụng ước lượng thí sinh đang nhìn vào
trong hay ra ngoài vùng màn hình (thi máy), hoặc liếc ngang kéo dài dù đầu không quay (cả hai kiểu).
Tín hiệu hướng mắt là tín hiệu thử nghiệm hỗ trợ rà soát, có mức điểm tối đa thấp.

**Why this priority**: Thí sinh có thể liếc mắt mà không quay đầu; quay đầu một mình bỏ sót hành vi này.
Nhưng ước lượng mắt qua webcam dễ sai, nên tín hiệu chỉ là hỗ trợ.

**Independent Test**: Sau hiệu chỉnh, thí sinh liếc mắt sang một bên ngoài màn hình 5 giây mà không quay
đầu; có một cảnh báo hướng mắt mức thấp. Nhìn vào các góc màn hình không tạo cảnh báo.

**Acceptance Scenarios**:

1. **Given** phiên thi máy chưa hiệu chỉnh hướng mắt, **When** chạy, **Then** không có cảnh báo hướng mắt
   và giao diện hiển thị "Hướng mắt: chưa hiệu chỉnh".
2. **Given** đã hiệu chỉnh, **When** thí sinh nhìn ra ngoài vùng màn hình quá thời lượng cấu hình,
   **Then** có cảnh báo "Nhìn ra ngoài màn hình" mức thấp, có độ tin cậy và chất lượng quan sát.
3. **Given** ảnh mắt chất lượng thấp (mặt nhỏ, nghiêng mạnh, đeo kính lóa, nhắm mắt), **When** đánh giá,
   **Then** tín hiệu bị bỏ qua thay vì đoán, và trạng thái "chất lượng thấp" hiển thị cho giám thị.
4. **Given** phiên thi giấy, **When** thí sinh nhìn xuống bài thi, **Then** không có cảnh báo hướng mắt.

---

### User Story 5 - Phát hiện nói chuyện (Priority: P2)

Ứng dụng nhận biết chuyển động miệng có nhịp giống nói kéo dài và tạo cảnh báo "Có thể đang nói" mức
thấp. Ngáp, há miệng tĩnh, nhai ngắn hoặc mấp máy đọc thầm ngắn không tạo cảnh báo.

**Why this priority**: Nói chuyện với người khác là hành vi gian lận phổ biến trong bộ dữ liệu tham
khảo, nhưng chuyển động miệng dễ báo nhầm nên là tín hiệu hỗ trợ.

**Independent Test**: Phát lại đoạn video thí sinh nói liên tục 5 giây; có một cảnh báo mức thấp. Đoạn
ngáp 3 giây không có cảnh báo.

**Acceptance Scenarios**:

1. **Given** thí sinh nói liên tục quá thời lượng cấu hình, **When** phát hiện, **Then** có một cảnh báo
   "Có thể đang nói" mức thấp, không bao giờ một mình đẩy mức rủi ro lên cao.
2. **Given** thí sinh ngáp hoặc há miệng giữ nguyên, **When** phát hiện, **Then** không có cảnh báo.
3. **Given** mặt bị che miệng hoặc chất lượng thấp, **When** đánh giá, **Then** tín hiệu bị bỏ qua.

---

### User Story 6 - Hoạt động tay đáng chú ý (Priority: P2)

Ứng dụng phát hiện tay đưa lên gần mặt hoặc tai kéo dài (ví dụ cầm tai nghe, che miệng nói) và tay di
chuyển liên tục ngoài vùng làm bài. Viết bài, gõ phím, chống cằm thông thường không tạo cảnh báo.

**Why this priority**: Bản hiện tại so sai tay giữa các frame và dùng vùng gần mặt quá rộng nên báo nhầm
nhiều.

**Independent Test**: Phát lại đoạn chống cằm 30 giây và đoạn viết bài 60 giây: không cảnh báo. Đoạn đưa
tay giữ tai 5 giây: một cảnh báo mức thấp.

**Acceptance Scenarios**:

1. **Given** hai tay trong khung hình đổi thứ tự giữa các frame, **When** đo chuyển động, **Then** chuyển
   động được tính theo từng tay thật, không tạo chuyển động giả.
2. **Given** thí sinh chống cằm, **When** phát hiện, **Then** không có cảnh báo tay gần mặt.
3. **Given** tay giữ gần tai quá thời lượng cấu hình, **When** phát hiện, **Then** có một cảnh báo mức thấp.

---

### User Story 7 - Rà soát và đánh dấu báo nhầm (Priority: P3)

Giám thị chọn một cảnh báo để xem đúng khung ảnh đã phân tích, khung bao vật thể, độ tin cậy, phiên bản
mô hình, kiểu bài thi và lý do. Giám thị đánh dấu "Báo nhầm" hoặc "Đã xem"; nhãn này được lưu cùng sự
kiện để làm dữ liệu cho lần chỉnh sau.

**Why this priority**: Hiến pháp yêu cầu mọi cảnh báo phải rà soát được; nhãn báo nhầm là nguồn dữ liệu
thật để cải thiện ngưỡng.

**Independent Test**: Mở một cảnh báo điện thoại, xem ảnh có khung bao, bấm "Báo nhầm"; mở lại tệp sự
kiện thấy nhãn rà soát với thời điểm.

**Acceptance Scenarios**:

1. **Given** một cảnh báo có ảnh bằng chứng, **When** mở chi tiết, **Then** thấy ảnh, khung bao, độ tin
   cậy, phiên bản mô hình, thời điểm bắt đầu và thời lượng.
2. **Given** chi tiết cảnh báo, **When** bấm "Báo nhầm", **Then** nhãn được lưu cùng sự kiện, cảnh báo
   hiển thị trạng thái đã rà soát, và điểm rủi ro hiện tại không bị đổi ngược trong lịch sử.

---

### Edge Cases

- Mô hình mặt không khả dụng: hướng đầu, hướng mắt và nói chuyện chuyển sang trạng thái "không khả dụng"
  hiển thị rõ; các luật dựa trên mặt không sinh cảnh báo.
- Ánh sáng yếu hoặc ngược sáng: tách "chất lượng ảnh thấp" khỏi "che camera"; lỗi camera hoặc mất frame
  được báo là sự cố thiết bị, không phải hành vi.
- Thí sinh đeo kính hoặc khẩu trang: hướng mắt hoặc nói chuyện báo "chất lượng thấp" thay vì đoán.
- Hiệu chỉnh thất bại (đầu di chuyển, không thấy mặt) trong thời gian giới hạn: báo thất bại, giữ trạng
  thái chưa hiệu chỉnh, cho phép thử lại.
- AI chạy chậm (khoảng cách giữa các lần phân tích lớn): luật thời gian không tự ghép hai lần quan sát
  cách xa thành một hành vi liên tục.
- Có thêm người chỉ thoáng qua phía sau dưới thời lượng cấu hình: không cảnh báo.
- Thay đổi kiểu bài thi giữa phiên: không cho phép; phải dừng và bắt đầu phiên mới.

## Requirements *(mandatory)*

### Functional Requirements

**Đánh giá và hồi quy**

- **FR-001**: Hệ thống MUST phát lại video đã ghi qua đúng chuỗi phân tích và luật như khi chạy webcam,
  với thời gian lấy theo dấu thời gian của video để kết quả lặp lại được.
- **FR-002**: Hệ thống MUST đọc file nhãn khoảng thời gian hành vi theo định dạng được tài liệu hóa và
  chấm điểm ở mức sự kiện: đúng, bỏ sót, báo nhầm, báo nhầm mỗi phút, độ trễ phát hiện, cho từng loại.
- **FR-003**: Hệ thống MUST lưu báo cáo đánh giá kèm cấu hình, phiên bản mô hình và thông tin máy chạy,
  và so sánh với một báo cáo mốc.
- **FR-004**: Bộ dữ liệu đánh giá MUST nằm ngoài kho mã, có tệp mô tả nguồn, giấy phép/điều kiện sử dụng
  và cách ánh xạ nhãn gốc sang loại sự kiện của hệ thống.

**Kiểu bài thi**

- **FR-005**: Giám thị MUST chọn kiểu bài thi "Thi giấy" hoặc "Thi máy" trước khi bắt đầu phiên; lựa chọn
  được ghi vào ảnh chụp chính sách của phiên và cố định trong phiên.
- **FR-006**: Mỗi kiểu bài thi MUST có bộ tham số luật riêng trong cấu hình (bật/tắt luật, ngưỡng, thời
  lượng), không viết điều kiện kiểu bài thi rải rác trong luật.
- **FR-007**: Thi giấy MUST không cảnh báo cúi đầu hoặc nhìn xuống bàn; thi máy MUST cảnh báo cúi đầu chỉ
  khi vượt thời lượng cấu hình riêng cho thi máy.

**Tín hiệu chính (kéo FR-12, FR-13, FR-14, FR-15, FR-17 của spec 001)**

- **FR-008**: Mọi luật MUST dùng đúng tham số thời lượng và ngưỡng của chính nó từ cấu hình; cấu hình
  chứa khóa không dùng hoặc thiếu khóa bắt buộc MUST bị từ chối khi khởi động với thông báo rõ.
- **FR-009**: Mọi ngưỡng đang viết cứng trong mã (chất lượng tối thiểu, khoảng hở mẫu, vùng gần mặt, ngưỡng
  chuyển động, ngưỡng mờ, vùng rời chỗ, mức điểm tối đa cho tín hiệu phụ) MUST chuyển vào cấu hình với
  mặc định được tài liệu hóa.
- **FR-010**: Một lần vật thể xuất hiện liên tục MUST chỉ mở một sự kiện; mất phát hiện ngắn hơn khoảng hở
  cấu hình hoặc đổi định danh theo dõi không được mở sự kiện thứ hai.
- **FR-011**: Hệ thống MUST áp thứ bậc tín hiệu: khi một tín hiệu gốc đang hoạt động (vắng người, che
  camera), các tín hiệu hệ quả (không thấy mặt, rời chỗ, hướng đầu, hướng mắt, nói chuyện) MUST được gắn
  làm tín hiệu phụ của sự kiện gốc và không cộng điểm rủi ro riêng.
- **FR-012**: Rời chỗ MUST dựa trên kết hợp vị trí người, mặt và vai so với vùng ngồi được ghi nhận lúc
  bắt đầu phiên, thay cho một ngưỡng vị trí cố định.
- **FR-013**: Che camera MUST dựa trên nhiều đặc trưng ảnh (độ sáng, độ tương phản, chi tiết cạnh, độ
  mờ) và MUST tách lỗi thiết bị hoặc mất frame khỏi hành vi.
- **FR-014**: Hiệu chỉnh hướng đầu MUST có đề xuất tự động khi quan sát ổn định, hiển thị trạng thái,
  từ chối mốc không chắc chắn và giữ thao tác hiệu chỉnh thủ công.
- **FR-015**: Mỗi cảnh báo vật thể MUST truy được về khung ảnh đã phân tích, khung bao, độ tin cậy, phiên
  bản mô hình, định danh theo dõi và thời điểm.
- **FR-016**: Lớp vật thể dùng để phát hiện tài liệu MUST chỉ gồm lớp mô hình thật sự hỗ trợ; giới hạn
  không phát hiện giấy rời MUST được ghi rõ trong tài liệu và giao diện.

**Hướng mắt (kéo FR-16 của spec 001)**

- **FR-017**: Hệ thống MUST ước lượng hướng mắt từ vị trí đồng tử so với khóe mắt, kết hợp hướng đầu, kèm
  chỉ số chất lượng quan sát (kích thước mắt, độ mở mắt, góc nghiêng đầu).
- **FR-018**: Hệ thống MUST cung cấp bước hiệu chỉnh hướng mắt: thi máy nhìn tâm và bốn góc màn hình; thi
  giấy nhìn tâm màn hình/giám thị và nhìn xuống bài. Hiệu chỉnh có thời hạn giới hạn, báo thành công
  hoặc thất bại.
- **FR-019**: Thi máy MUST tạo sự kiện "Nhìn ra ngoài màn hình" khi hướng mắt nằm ngoài vùng hiệu chỉnh
  cộng biên cấu hình quá thời lượng cấu hình; cả hai kiểu MUST tạo sự kiện "Liếc ngang kéo dài" khi mắt
  lệch ngang mạnh mà đầu không quay.
- **FR-020**: Tín hiệu hướng mắt MUST là tín hiệu thử nghiệm hỗ trợ, có mức điểm tối đa cấu hình được,
  tắt hoặc bật được qua cấu hình, và bị bỏ qua khi chất lượng quan sát dưới ngưỡng.

**Nói chuyện (kéo FR-16 của spec 001)**

- **FR-021**: Hệ thống MUST đo độ mở miệng theo thời gian, chuẩn hóa theo kích thước mặt, và nhận biết
  dao động có nhịp giống nói trong một cửa sổ thời gian cấu hình.
- **FR-022**: Há miệng giữ nguyên (ngáp) hoặc dao động quá ngắn MUST không tạo sự kiện.
- **FR-023**: Tín hiệu nói chuyện MUST là tín hiệu hỗ trợ có mức điểm tối đa, bị bỏ qua khi miệng bị che
  hoặc chất lượng quan sát thấp.

**Hoạt động tay**

- **FR-024**: Chuyển động tay MUST được đo theo từng tay đã ghép đúng giữa các lần phân tích, chuẩn hóa theo
  thời gian giữa hai lần và kích thước mặt.
- **FR-025**: "Tay gần mặt/tai" MUST dựa trên điểm mốc của mặt và tay, có thời lượng tối thiểu; chống cằm
  thông thường không thỏa luật.

**Rủi ro và tương quan (kéo FR-18, FR-19 của spec 001)**

- **FR-026**: Tương quan chỉ ghép sự kiện đã xác nhận, cùng phiên, còn mới, từ bộ phân tích khỏe; chống
  cộng trùng, có giới hạn và liên kết sự kiện nguồn.
- **FR-027**: Điểm rủi ro hiện tại MUST giảm dần khi hành vi dừng; số sự kiện, điểm tích lũy và điểm đỉnh
  chỉ là lịch sử, không thành mức sàn của điểm hiện tại.

**Rà soát**

- **FR-028**: Giám thị MUST xem chi tiết cảnh báo (ảnh, khung bao, độ tin cậy, phiên bản mô hình, kiểu bài
  thi, lý do) và gắn nhãn "Báo nhầm" hoặc "Đã xem"; nhãn được lưu kèm thời điểm.

**Chất lượng cấu trúc**

- **FR-029**: Luật hành vi, thứ bậc tín hiệu và tính điểm MUST nằm trong lớp nghiệp vụ, kiểm thử được mà
  không cần camera, mô hình hay giao diện; lớp ứng dụng desktop chỉ điều phối và hiển thị.
- **FR-030**: Dữ liệu trao đổi giữa lớp phân tích, lớp nghiệp vụ và lớp hiển thị MUST có kiểu được khai báo
  và kiểm tra tĩnh được.
- **FR-031**: Mỗi bước tái cấu trúc không chủ đích đổi hành vi MUST giữ nguyên kết quả phát lại so với mốc
  ghi trước khi tái cấu trúc.
- **FR-032**: Kiểm tra mã (lint, kiểm tra kiểu, kiểm thử) MUST đạt 0 lỗi trên nhánh trước khi hoàn tất.

### Key Entities

- **Kiểu bài thi (Exam mode)**: "paper" hoặc "computer"; quyết định bộ tham số luật; thuộc ảnh chụp chính
  sách của phiên.
- **Quan sát (Observation)**: kết quả một lần phân tích một frame: người, mặt (hướng đầu, hướng mắt, độ
  mở miệng, chất lượng), tay, vật thể, chất lượng ảnh, sức khỏe từng bộ phân tích.
- **Tín hiệu luật (Rule signal)**: trạng thái của một luật tại một thời điểm: loại, ứng viên/hoạt động/kết
  thúc, độ tin cậy, chỉ số, mã lý do, mức điểm tối đa.
- **Sự kiện (Suspicious event)**: một episode hành vi có bắt đầu, kết thúc, thời lượng, mức độ, sự kiện gốc
  (nếu là tín hiệu phụ), bằng chứng và nhãn rà soát.
- **Hồ sơ hiệu chỉnh (Calibration profile)**: mốc hướng đầu, vùng hướng mắt, vùng ngồi; có trạng thái
  (chưa có, đang thu, thành công, thất bại) và thời điểm.
- **Bộ đánh giá (Evaluation set)**: danh sách video, nhãn khoảng thời gian, ánh xạ nhãn, nguồn và giấy phép.
- **Báo cáo đánh giá (Evaluation report)**: chỉ số theo loại sự kiện, cấu hình, mô hình, phần cứng, so
  sánh với mốc.
- **Nhãn rà soát (Review label)**: "false_positive" hoặc "reviewed", người gắn (cục bộ), thời điểm.

## Success Criteria *(mandatory)*

Các số tuyệt đối được chốt sau khi có báo cáo mốc (US1); tiêu chí dưới đây đo tương đối với mốc đó.

### Measurable Outcomes

- **SC-001**: Phát lại cùng video và cấu hình hai lần cho danh sách sự kiện giống nhau 100%.
- **SC-002**: Trên các đoạn ngồi làm bài bình thường của bộ đánh giá, báo nhầm mỗi phút của các tín hiệu
  chính giảm ít nhất 50% so với mốc.
- **SC-003**: Độ phủ (tỉ lệ khoảng hành vi có nhãn được cảnh báo) của mỗi tín hiệu chính không giảm quá 5
  điểm phần trăm so với mốc; điện thoại và vắng người không giảm.
- **SC-004**: Mỗi khoảng hành vi liên tục có nhãn tạo không quá 1 sự kiện chính (không trùng lặp) trong ít
  nhất 95% khoảng.
- **SC-005**: Ở thi giấy, đoạn cúi nhìn bài không tạo cảnh báo cúi đầu (0 cảnh báo trên bộ đánh giá).
- **SC-006**: Hướng mắt và nói chuyện có số đo độ phủ và báo nhầm mỗi phút trong báo cáo; không lúc nào một
  mình hai tín hiệu này đưa mức rủi ro vượt mức "trung bình".
- **SC-007**: Hiệu chỉnh (hướng đầu + hướng mắt) hoàn tất trong dưới 20 giây với thí sinh hợp tác.
- **SC-008**: Tốc độ AI trên máy tham chiếu không giảm quá 15% so với mốc sau khi thêm hướng mắt và nói
  chuyện.
- **SC-009**: Kiểm tra mã đạt 0 lỗi lint, 0 lỗi kiểu, toàn bộ kiểm thử qua.

## Assumptions

- Bộ đánh giá chính là OEP dataset (Michigan State University, Atoum et al., 2017), tải về
  `E:\datasets\oep`, chỉ dùng đánh giá nội bộ, trích dẫn bài báo gốc và không đưa vào Git. Bộ này quay
  người thật làm bài trên máy tính nên phản ánh chủ yếu kiểu "Thi máy".
- Các ca không có trong OEP (thi giấy, bộ bài/ví như ca âm tính khó, che camera) được bổ sung bằng clip
  do người dùng tự quay theo kịch bản trong quickstart; khi chưa có, báo cáo ghi rõ ca nào chưa được đo.
- Tiếp tục dùng các mô hình đã có trong manifest; mô hình mặt hiện tại cung cấp điểm mốc mống mắt và hệ
  số biểu cảm miệng, đủ cho hướng mắt và nói chuyện mà không thêm mô hình mới.
- Phase 1 vẫn là một camera, một thí sinh; không nhận dạng danh tính, cảm xúc hay nói dối.
- Mức sàn điểm rủi ro khi sự kiện đang hoạt động (mô tả trong README) được giữ: nó chỉ có hiệu lực khi
  sự kiện còn được cập nhật và hết sau thời hạn cấu hình, nên không mâu thuẫn FR-027.
- Đóng gói bộ cài `.exe` nằm ngoài phạm vi spec này.
