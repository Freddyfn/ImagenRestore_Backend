import cv2
import numpy as np
import base64
import time

class OpenCVImageProcessor:
    def apply_filters(self, image_data: bytes, method: str, tolerance: float, max_iterations: int) -> dict:
        start_time = time.time()
        np_arr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # ============================================================
        # PIPELINE: Convertir a LAB y procesar SOLO el canal L
        # Esto preserva los colores originales (canales a,b intactos)
        # y aplica la inversión de difusión únicamente a la luminosidad.
        # ============================================================
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        
        # Canal L como float32 para el solver iterativo
        f = l_ch.astype(np.float32)
        H, W = f.shape[:2]
        
        # ============================================================
        # MODELO MATEMÁTICO: Deconvolución Iterativa (Inversión de Difusión)
        #
        # Problema: La imagen f está degradada (borrosa, con ruido suavizado).
        # Modelo:   f = (I + α∇²) u_sharp
        #           La imagen original nítida u sufrió una difusión de calor.
        #
        # Solución: Resolver el sistema lineal (I + α∇²) u = f
        #
        # Discretización (5-point stencil):
        #   (1 - 4α) u_{i,j} + α (u_N + u_S + u_W + u_E) = f_{i,j}
        #
        # Jacobi:
        #   u_{i,j}^{new} = [ f_{i,j} - α(u_N + u_S + u_W + u_E) ] / (1 - 4α)
        #
        # EFECTO: Invierte la difusión. Los bordes se AMPLIFICAN porque
        #         el denominador (1-4α) < 1 escala los valores hacia arriba
        #         en zonas de alto gradiente.
        #
        # Estabilidad: α < 0.125 garantiza diagonal dominante → convergencia.
        # ============================================================
        
        alpha = 0.11  # Parámetro de inversión de difusión (α < 0.125 para estabilidad)
        diag = 1.0 - 4.0 * alpha  # Elemento diagonal de la matriz del sistema
        
        # Inicialización: la imagen degradada como primera aproximación
        u = np.copy(f)
        u_old = np.empty_like(u)
        
        # Red-Black masks para vectorización de Gauss-Seidel/SOR
        y, x = np.mgrid[0:H-2, 0:W-2]
        mask_red = (x + y) % 2 == 0
        mask_black = (x + y) % 2 == 1
        
        # Cálculo del Omega Óptimo
        # Radio Espectral de Jacobi para el sistema (I + α∇²):
        cos_max = 0.5 * (np.cos(np.pi / H) + np.cos(np.pi / W))
        rho_jacobi = (4.0 * alpha / abs(diag)) * cos_max
        if rho_jacobi < 1.0:
            omega_opt = 2.0 / (1.0 + np.sqrt(1.0 - rho_jacobi**2))
        else:
            omega_opt = 1.0
        
        chart_data = []
        final_error = 0.0
        iters_done = 0
        
        for k in range(max_iterations):
            u_old[:] = u
            
            if method == 'jacobi':
                neighbors = u_old[:-2, 1:-1] + u_old[2:, 1:-1] + u_old[1:-1, :-2] + u_old[1:-1, 2:]
                u[1:-1, 1:-1] = (f[1:-1, 1:-1] - alpha * neighbors) / diag
                
            elif method == 'gauss-seidel' or method == 'sor':
                w = 1.0 if method == 'gauss-seidel' else omega_opt
                
                # Red update
                neighbors_r = u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:]
                u_gs_r = (f[1:-1, 1:-1] - alpha * neighbors_r) / diag
                u[1:-1, 1:-1][mask_red] = (1 - w) * u[1:-1, 1:-1][mask_red] + w * u_gs_r[mask_red]
                
                # Black update
                neighbors_b = u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:]
                u_gs_b = (f[1:-1, 1:-1] - alpha * neighbors_b) / diag
                u[1:-1, 1:-1][mask_black] = (1 - w) * u[1:-1, 1:-1][mask_black] + w * u_gs_b[mask_black]
                
            elif method == 'ssor':
                w = omega_opt
                
                # Forward Sweep: Red → Black
                neighbors = u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:]
                u_gs = (f[1:-1, 1:-1] - alpha * neighbors) / diag
                u[1:-1, 1:-1][mask_red] = (1 - w) * u[1:-1, 1:-1][mask_red] + w * u_gs[mask_red]
                
                neighbors = u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:]
                u_gs = (f[1:-1, 1:-1] - alpha * neighbors) / diag
                u[1:-1, 1:-1][mask_black] = (1 - w) * u[1:-1, 1:-1][mask_black] + w * u_gs[mask_black]
                
                # Backward Sweep: Black → Red
                neighbors = u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:]
                u_gs = (f[1:-1, 1:-1] - alpha * neighbors) / diag
                u[1:-1, 1:-1][mask_black] = (1 - w) * u[1:-1, 1:-1][mask_black] + w * u_gs[mask_black]
                
                neighbors = u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:]
                u_gs = (f[1:-1, 1:-1] - alpha * neighbors) / diag
                u[1:-1, 1:-1][mask_red] = (1 - w) * u[1:-1, 1:-1][mask_red] + w * u_gs[mask_red]
            
            # Cálculo del error de convergencia
            diff = np.abs(u - u_old)
            current_error = float(np.max(diff))
            
            chart_data.append({"iteration": k, "error": current_error})
            
            iters_done = k + 1
            final_error = current_error
            
            if current_error < tolerance:
                break
        
        # Clip al rango válido de L (0-255 en OpenCV LAB)
        l_sharp = np.clip(u, 0, 255).astype(np.uint8)
        
        # CLAHE suave sobre el canal L ya afilado (1.4 para darle un poco más de 'punch')
        clahe = cv2.createCLAHE(clipLimit=1.4, tileGridSize=(8, 8))
        l_final = clahe.apply(l_sharp)
        
        # Reconstruir imagen: L procesado + a,b ORIGINALES intactos
        lab_final = cv2.merge((l_final, a_ch, b_ch))
        result = cv2.cvtColor(lab_final, cv2.COLOR_LAB2BGR)
        
        _, buffer = cv2.imencode('.png', result)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        end_time = time.time()
        elapsed = round(end_time - start_time, 2)
        
        return {
            "restored_image_base64": f"data:image/png;base64,{img_base64}",
            "metrics": {
                "time_seconds": elapsed,
                "final_error": round(final_error, 6),
                "iterations": iters_done,
                "method": method
            },
            "chart_data": chart_data
        }
