import cv2
import time

class SOPMonitoring:
    def __init__(self):
        self.sop_sequence = [
            {"get [Clamp Bracket]", "get [L-Shaped Bracket]"},
            "get screw [1]",
            "get [Allen Key]",
            "Put [Allen Key]",
            "get [Sliding Connector]",
            "get screw [2]",
            "get [Allen Key]",
            "Put [Allen Key]",
            "get [Connector Block]",
            "get screw [3]",
            "get [Allen Key]",
            "Put [Allen Key]",
        ]
        self.current_step = 0
        self.error_message = ""
        self.last_error_frame = None  # Lưu frame lỗi cuối cùng

    def reset_monitoring(self):
        """Reset lại quá trình SOP sau khi hiển thị lỗi"""
        print("[RESET] Restarting SOP monitoring...")
        self.current_step = 0
        self.error_message = ""
        self.last_error_frame = None
        time.sleep(2)  # Đợi 2 giây trước khi bắt đầu lại

    def validate_action(self, detected_action):
        """Xác thực hành động hiện tại"""
        expected_action = self.sop_sequence[self.current_step]
        
        if isinstance(expected_action, set):
            if detected_action in expected_action:
                expected_action.remove(detected_action)
                if not expected_action:  # Nếu tất cả các hành động trong set đã xong, chuyển bước tiếp theo
                    self.current_step += 1
                return True
        elif detected_action == expected_action:
            self.current_step += 1
            return True
        
        self.error_message = f"[ERROR] Incorrect action: {detected_action}. Expected: {expected_action}"
        return False

    def display_error(self, frame):
        """Hiển thị lỗi lên frame video"""
        if self.error_message:
            cv2.putText(frame, self.error_message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    def process_frame(self, frame, detected_action):
        """Xử lý từng frame và hiển thị lỗi nếu cần"""
        if self.validate_action(detected_action):
            print(f"[VALID] Action: {detected_action}. Moving to next step.")
        else:
            print(self.error_message)
            self.last_error_frame = frame.copy()  # Lưu lại frame cuối khi lỗi
            self.display_error(self.last_error_frame)

            # Giữ màn hình lỗi trong 2 giây trước khi reset
            cv2.imshow("SOP Monitoring", self.last_error_frame)
            cv2.waitKey(2000)

            self.reset_monitoring()
