# scripts/migrate_role_meta.py
import os
import json


def migrate_role_metadata(base_dir="../data/roles"):
    """
    无停机数据库迁移脚本：
    将历史角色数据中被错误嵌套的 system_prompt 提取到根节点。
    """
    if not os.path.exists(base_dir):
        print(f"目录不存在: {base_dir}")
        return

    for role_folder in os.listdir(base_dir):
        meta_path = os.path.join(base_dir, role_folder, "role_meta.json")
        if not os.path.isfile(meta_path):
            continue

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            needs_update = False

            # 检查是否发生了层级倒置 (System Prompt 在 settings 内部)
            if "settings" in data and isinstance(data["settings"], dict):
                if "system_prompt" in data["settings"]:
                    # 核心修复：执行提取与删除，将提示词提升至根层级
                    data["system_prompt"] = data["settings"].pop("system_prompt")
                    needs_update = True

            if needs_update:
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                print(f"✅ 成功迁移角色数据拓扑: {role_folder}")
            else:
                print(f"⚡ 角色数据已是最新同级拓扑，无需迁移: {role_folder}")

        except Exception as e:
            print(f"❌ 迁移角色 {role_folder} 失败: {e}")


if __name__ == "__main__":
    print("开始执行角色元数据拓扑迁移...")
    migrate_role_metadata()