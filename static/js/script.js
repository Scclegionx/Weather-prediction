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

  // Ẩn tab chi tiết ban đầu
  detailTab.style.display = "none";

  // Thêm sự kiện khi di chuột vào ngày
  day.addEventListener("mouseenter", () => {
    // Hiển thị tab chi tiết
    day.appendChild(detailTab);
    detailTab.style.display = "block";
  });

  // Thêm sự kiện khi di chuột ra khỏi ngày
  day.addEventListener("mouseleave", () => {
    // Ẩn tab chi tiết
    detailTab.style.display = "none";
  });
});
