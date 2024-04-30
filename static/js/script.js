// Lấy danh sách các thẻ chứa thông tin ngày
const forecastDays = document.querySelectorAll(".forecast-day");

// Lặp qua từng ngày và thêm sự kiện khi di chuột vào
forecastDays.forEach((day) => {
  // Lấy tên ngày và các thông tin cần hiển thị
  const dayName = day.querySelector(".date").textContent;
  const maxTemp = day.querySelector(".high").textContent;
  const minTemp = day.querySelector(".low").textContent;
  const description = day.querySelector(".description").textContent;

  // Tạo tab chứa thông tin chi tiết
  const detailTab = document.createElement("div");
  detailTab.classList.add("detail-tab");
  detailTab.innerHTML = `
    <div>Max Temp: ${maxTemp}</div>
    <div>Min Temp: ${minTemp}</div>
    <div>Description: ${description}</div>
    <div>Humidity: 80%</div> <!-- Thay bằng giá trị thực tế nếu có -->
    <div>Wind Direction: North</div> <!-- Thay bằng giá trị thực tế nếu có -->
  `;

  // Biến để theo dõi xem con trỏ chuột có đang ở trên tab chi tiết hay không
  let isMouseOverDetailTab = false;

  // Thêm sự kiện khi di chuột vào ngày
  day.addEventListener("mouseenter", () => {
    // Lấy kích thước của ô ngày
    const dayWidth = day.offsetWidth;
    const dayHeight = day.offsetHeight;

    // Lấy kích thước của tab chi tiết
    const tabWidth = detailTab.offsetWidth;
    const tabHeight = detailTab.offsetHeight;

    // Tính toán vị trí để tab hiển thị ở giữa so với ô ngày
    const offsetX = day.offsetLeft + (dayWidth - tabWidth) / 2;
    const offsetY = day.offsetTop + dayHeight;

    // Đặt vị trí cho tab
    detailTab.style.left = offsetX + "px";
    detailTab.style.top = offsetY + "px";

    // Thêm tab vào body và hiển thị
    document.body.appendChild(detailTab);
    detailTab.style.display = "block";
  });

  // Thêm sự kiện khi di chuột ra khỏi ngày hoặc tab chi tiết
  day.addEventListener("mouseleave", () => {
    // Ẩn và xóa tab chi tiết
    detailTab.style.display = "none";
    if (detailTab.parentNode) {
      detailTab.parentNode.removeChild(detailTab);
    }
  });

  // Thêm sự kiện khi di chuột vào tab chi tiết
  detailTab.addEventListener("mouseenter", () => {
    isMouseOverDetailTab = true;
  });

  // Thêm sự kiện khi di chuột ra khỏi tab chi tiết
  detailTab.addEventListener("mouseleave", () => {
    isMouseOverDetailTab = false;
    // Ẩn và xóa tab chi tiết
    detailTab.style.display = "none";
    if (detailTab.parentNode) {
      detailTab.parentNode.removeChild(detailTab);
    }
  });
});
