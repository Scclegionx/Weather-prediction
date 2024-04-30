// Lấy danh sách các thẻ chứa thông tin ngày
const forecastDays = document.querySelectorAll(".forecast-day");

// Lặp qua từng ngày và thêm sự kiện khi di chuột vào
forecastDays.forEach((day) => {
  console.log(day)
  // Lấy tên ngày và các thông tin cần hiển thị
  const maxTemp = day.querySelector(".high").textContent;
  const minTemp = day.querySelector(".low").textContent;
  const humidity = day.querySelector(".humidity").textContent;

  // Tạo tab chứa thông tin chi tiết
  const detailTab = document.createElement("div");
  detailTab.classList.add("detail-tab");
  detailTab.innerHTML = `
    <div>${maxTemp}</div>
    <div>${minTemp}</div>
    <div>${humidity}</div> 
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