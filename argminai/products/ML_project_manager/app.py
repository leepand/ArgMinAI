import os
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    jsonify,
    flash,
    send_file,
    session,
)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from datetime import datetime
import difflib
import json

from pathlib import Path

# from flask import Markup
from markupsafe import Markup
import markdown

import pygments
from pygments.lexers import get_lexer_for_filename
from pygments.formatters import HtmlFormatter
from sqlalchemy.orm import Session
from flask_wtf.csrf import CSRFProtect

# 在创建 app 后初始化
app = Flask(__name__)
csrf = CSRFProtect(app)  # 添加这行
app.config.from_pyfile("config.py")
app.config["SECRET_KEY"] = "your-secret-key-here"


# 数据库配置
# 数据库配置 - 更新这部分
# 数据库配置 - 更新这部分
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///mlprojects.db"
db = SQLAlchemy(app)

DATABASE_DIR = Path("instance")
CODE_UPLOAD_FOLDER = Path("static/uploads/code")
# 文件上传配置
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"py", "ipynb", "txt", "md", "pdf", "json"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# 确保上传目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, SelectField, DateTimeField, StringField
from wtforms.validators import DataRequired, Optional
from wtforms.fields import DateField


class SegmentForm(FlaskForm):
    name = StringField("Segment Name", validators=[DataRequired()])
    description = TextAreaField("Description")
    project_id = SelectField("Project", coerce=int, validators=[Optional()])
    filter_rules = TextAreaField("Filter Rules", validators=[DataRequired()])


project_tags = db.Table(
    "project_tags",
    db.Column("project_id", db.Integer, db.ForeignKey("project.id"), primary_key=True),
    db.Column("tag_id", db.Integer, db.ForeignKey("tag.id"), primary_key=True),
)


class CampaignForm(FlaskForm):
    name = StringField("活动名称", validators=[DataRequired()])
    description = TextAreaField("活动描述")
    segment_id = SelectField("目标分群", coerce=int, validators=[DataRequired()])
    strategy = TextAreaField("策略配置")

    # 修改为日期选择器字段
    start_date = DateField("开始日期", format="%Y-%m-%d", validators=[DataRequired()])
    start_time = SelectField(
        "开始时间",
        choices=[
            ("09:00", "09:00 AM"),
            ("10:00", "10:00 AM"),
            # 添加更多时间选项...
        ],
        validators=[DataRequired()],
    )

    end_date = DateField("结束日期", format="%Y-%m-%d", validators=[DataRequired()])
    end_time = SelectField(
        "结束时间",
        choices=[
            ("17:00", "05:00 PM"),
            ("18:00", "06:00 PM"),
            # 添加更多时间选项...
        ],
        validators=[DataRequired()],
    )


# 数据库模型
class User(db.Model):
    # 保持原有字段完全不变 ▼▼▼
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(50), default="member")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<User {self.username}>"


class Project(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    owner_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    status = db.Column(db.String(50), default="active")

    owner = db.relationship("User", backref=db.backref("projects", lazy=True))
    tags = db.relationship("Tag", secondary=project_tags, backref="projects")

    def __repr__(self):
        return f"<Project {self.name}>"


class ProjectMember(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey("project.id"), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    role = db.Column(db.String(50), default="member")
    joined_at = db.Column(db.DateTime, default=datetime.utcnow)

    project = db.relationship("Project", backref=db.backref("members", lazy=True))
    user = db.relationship("User", backref=db.backref("project_memberships", lazy=True))


class CodeVersion(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey("project.id"), nullable=False)
    version = db.Column(db.String(50), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)  # 增加长度以适应长路径
    commit_message = db.Column(db.Text)
    author_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # 新增字段
    file_count = db.Column(db.Integer, default=0)
    total_size = db.Column(db.BigInteger, default=0)  # 使用BigInteger存储大文件尺寸

    # 关系
    project = db.relationship("Project", backref=db.backref("code_versions", lazy=True))
    author = db.relationship(
        "User", backref=db.backref("code_contributions", lazy=True)
    )


class DocumentVersion(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    is_draft = db.Column(db.Boolean, default=False)
    draft_key = db.Column(db.String(100))
    project_id = db.Column(db.Integer, db.ForeignKey("project.id"), nullable=False)
    version = db.Column(db.String(50), nullable=False, default="1.0.0")  # 添加默认值
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text)
    author_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    project = db.relationship("Project", backref=db.backref("documents", lazy=True))
    author = db.relationship("User", backref=db.backref("authored_docs", lazy=True))


class Experiment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey("project.id"), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    parameters = db.Column(db.Text)  # JSON格式存储
    metrics = db.Column(db.Text)  # JSON格式存储
    status = db.Column(db.String(50), default="running")
    created_by = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    project = db.relationship("Project", backref=db.backref("experiments", lazy=True))
    creator = db.relationship("User", backref=db.backref("experiments", lazy=True))


class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey("project.id"), nullable=False)
    experiment_id = db.Column(db.Integer, db.ForeignKey("experiment.id"), nullable=True)
    content = db.Column(db.Text, nullable=False)
    rating = db.Column(db.Integer)  # 1-5
    submitted_by = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    submitted_at = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(50), default="new")

    project = db.relationship("Project", backref=db.backref("feedbacks", lazy=True))
    experiment = db.relationship(
        "Experiment", backref=db.backref("feedbacks", lazy=True)
    )
    submitter = db.relationship("User", backref=db.backref("feedbacks", lazy=True))


# 在User模型后添加Tag模型和关联表
class Tag(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)
    color = db.Column(db.String(20))  # Tailwind颜色类
    icon = db.Column(db.String(30))  # FontAwesome图标类
    category = db.Column(db.String(20), default="general")  # 新增分类字段
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Tag {self.name}>"


# 特征管理
class Feature(db.Model):
    __tablename__ = "feature"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    short_description = db.Column(db.String(200))  # 新增：简短描述
    long_description = db.Column(db.Text)  # 原description改为长描述
    data_type = db.Column(db.String(50))
    project_id = db.Column(db.Integer, db.ForeignKey("project.id"))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # 关系
    project = db.relationship("Project", backref=db.backref("features", lazy=True))
    tags = db.relationship("Tag", secondary="feature_tags")


class FeatureVersion(db.Model):
    __tablename__ = "feature_version"
    id = db.Column(db.Integer, primary_key=True)
    feature_id = db.Column(db.Integer, db.ForeignKey("feature.id"))
    version = db.Column(db.String(50))
    sample_data = db.Column(db.Text)  # JSON格式示例数据
    schema_definition = db.Column(db.Text)  # JSON Schema
    created_by = db.Column(db.Integer, db.ForeignKey("user.id"))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # 关系
    feature = db.relationship("Feature", backref=db.backref("versions", lazy=True))
    author = db.relationship("User")


feature_tags = db.Table(
    "feature_tags",
    db.Column("feature_id", db.Integer, db.ForeignKey("feature.id")),
    db.Column("tag_id", db.Integer, db.ForeignKey("tag.id")),
)


# 在现有模型后添加以下模型
class UserSegment(db.Model):
    """用户分群模型"""

    __tablename__ = "user_segment"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    project_id = db.Column(db.Integer, db.ForeignKey("project.id"), nullable=True)
    filter_rules = db.Column(db.Text)  # JSON格式的过滤规则
    created_by = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # 明确指定关系
    project = db.relationship("Project", backref=db.backref("segments", lazy=True))
    creator = db.relationship("User", backref=db.backref("created_segments", lazy=True))

    # 修复多对多关系
    users = db.relationship(
        "User",
        secondary="segment_users",
        primaryjoin="UserSegment.id == SegmentUser.segment_id",
        secondaryjoin="User.id == SegmentUser.user_id",
        backref=db.backref("segments", lazy=True),
    )

    campaigns = db.relationship("Campaign", backref=db.backref("segment", lazy=True))


class SegmentUser(db.Model):
    """分群用户关联表"""

    __tablename__ = "segment_users"

    segment_id = db.Column(
        db.Integer, db.ForeignKey("user_segment.id"), primary_key=True
    )
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), primary_key=True)
    added_at = db.Column(db.DateTime, default=datetime.utcnow)
    added_by = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True)

    # 明确指定关系
    adder = db.relationship("User", foreign_keys=[added_by])


class Campaign(db.Model):
    """营销活动模型"""

    __tablename__ = "campaign"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    segment_id = db.Column(db.Integer, db.ForeignKey("user_segment.id"), nullable=False)
    strategy = db.Column(db.Text)  # JSON格式的策略配置
    start_time = db.Column(db.DateTime)
    end_time = db.Column(db.DateTime)
    status = db.Column(
        db.String(20), default="draft"
    )  # draft, active, completed, archived
    created_by = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # 关系
    creator = db.relationship("User", backref=db.backref("campaigns", lazy=True))
    feedbacks = db.relationship(
        "CampaignFeedback", backref=db.backref("campaign", lazy=True)
    )


class CampaignFeedback(db.Model):
    """营销活动反馈"""

    __tablename__ = "campaign_feedback"

    id = db.Column(db.Integer, primary_key=True)
    campaign_id = db.Column(db.Integer, db.ForeignKey("campaign.id"), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    rating = db.Column(db.Integer)  # 1-5
    feedback = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # 关系
    user = db.relationship("User", backref=db.backref("campaign_feedbacks", lazy=True))


# 辅助函数
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_diff(old_text, new_text):
    diff = difflib.unified_diff(
        old_text.splitlines(keepends=True),
        new_text.splitlines(keepends=True),
        fromfile="old",
        tofile="new",
        lineterm="",
    )
    return "".join(diff)


# 初始化数据库
def initialize_database():
    # 初始化数据库
    with app.app_context():
        try:
            db.create_all()
            print(f"Database created at: mlprojects.db")
            # 创建示例标签（如果不存在）
            if not Tag.query.first():
                game_ml_tags = [
                    # 算法方向
                    {
                        "name": "Reinforcement Learning",
                        "color": "purple",
                        "category": "algorithm",
                    },
                    {
                        "name": "Multi-Agent Systems",
                        "color": "fuchsia",
                        "category": "algorithm",
                    },
                    {
                        "name": "Imitation Learning",
                        "color": "violet",
                        "category": "algorithm",
                    },
                    {
                        "name": "Player Modeling",
                        "color": "blue",
                        "category": "algorithm",
                    },
                    # 技术框架
                    {
                        "name": "Unity ML-Agents",
                        "color": "orange",
                        "category": "framework",
                    },
                    {"name": "PyTorch", "color": "red", "category": "framework"},
                    {"name": "Ray RLlib", "color": "lime", "category": "framework"},
                    # 游戏应用
                    {"name": "NPC Behavior", "color": "teal", "category": "game"},
                    {"name": "Dynamic Difficulty", "color": "cyan", "category": "game"},
                    {"name": "Matchmaking", "color": "sky", "category": "game"},
                    # 核心组件
                    {
                        "name": "Reward Design",
                        "color": "yellow",
                        "category": "component",
                    },
                    {
                        "name": "Policy Network",
                        "color": "amber",
                        "category": "component",
                    },
                ]

                for tag_data in game_ml_tags:
                    tag = Tag(
                        name=tag_data["name"],
                        color=tag_data["color"],
                        category=tag_data.get("category", "general"),
                    )
                    db.session.add(tag)

                db.session.commit()
                print("Initialized game ML tags")

            # 创建示例用户（如果数据库为空）
            if not User.query.first():
                admin = User(
                    username="admin",
                    email="admin@example.com",
                    password="password",  # 在实际应用中应该使用哈希密码
                    role="admin",
                )
                db.session.add(admin)

                # 创建示例项目
                project = Project(
                    name="Sentiment Analysis",
                    description="Analyze customer reviews sentiment",
                    owner_id=1,
                    status="active",
                )
                db.session.add(project)

                db.session.commit()
                print("Created initial admin user and sample project")
        except Exception as e:
            print(f"Error initializing database: {e}")
            # 确保数据库文件可写
            db_file = DATABASE_DIR / "app.db"
            if db_file.exists():
                print(
                    f"Checking database file permissions: {os.access(db_file, os.W_OK)}"
                )
            raise e


# 手动初始化或使用@app.cli.command()
initialize_database()  # 直接初始化，或使用下面的命令行方式


# 或者添加命令行初始化方式（推荐）
@app.cli.command("init-db")
def init_db_command():
    """Initialize the database."""
    initialize_database()
    print("Initialized the database.")


# 修改渲染过滤器
@app.template_filter("markdown")
def markdown_filter(text):
    import markdown

    extensions = [
        "fenced_code",  # 代码块
        "tables",  # 表格
        "footnotes",  # 脚注
        "toc",  # 目录
        "nl2br",  # 换行转<br>
        "sane_lists",  # 更合理的列表
        "mdx_math",  # 数学公式
        "pymdownx.highlight",  # 代码高亮
        "pymdownx.superfences",  # 更好的代码块
    ]
    return Markup(markdown.markdown(text, extensions=extensions))


# 在 app.py 中添加以下代码
@app.context_processor
def utility_processor():
    def get_code_files(project):
        """获取项目代码文件结构"""
        files = []
        if project and project.code_versions:
            try:
                latest_version = project.code_versions[-1]
                code_dir = os.path.dirname(latest_version.file_path)

                for root, dirs, filenames in os.walk(code_dir):
                    for dirname in dirs:
                        files.append(
                            {
                                "name": dirname,
                                "path": os.path.join(root, dirname),
                                "is_dir": True,
                            }
                        )

                    for filename in filenames:
                        files.append(
                            {
                                "name": filename,
                                "path": os.path.join(root, filename),
                                "is_dir": False,
                            }
                        )
            except Exception as e:
                print(f"Error getting code files: {e}")
                return []

        return files

    return dict(get_code_files=get_code_files)


# 路由定义
# 项目路由
@app.route("/")
def index():
    return redirect(url_for("project_list"))


@app.route("/projects")
def project_list():
    projects = Project.query.all()
    return render_template("projects/list.html", projects=projects)


@app.route("/projects/<int:project_id>")
def project_detail(project_id):
    project = Project.query.get_or_404(project_id)
    return render_template("projects/detail.html", project=project)


@app.route("/projects/create", methods=["GET", "POST"])
def project_create():
    all_tags = Tag.query.all()

    if request.method == "POST":
        name = request.form.get("name")
        description = request.form.get("description")

        new_project = Project(
            name=name,
            description=description,
            owner_id=1,  # 假设当前用户ID为1
            status="active",
        )

        # 添加选择的标签
        selected_tag_ids = request.form.getlist("tags")
        new_project.tags = Tag.query.filter(Tag.id.in_(selected_tag_ids)).all()

        db.session.add(new_project)
        db.session.commit()
        return redirect(url_for("project_detail", project_id=new_project.id))

    return render_template("projects/form.html", all_tags=all_tags, is_edit=False)


@app.route("/projects/<int:project_id>/edit", methods=["GET", "POST"])
def project_edit(project_id):
    project = Project.query.get_or_404(project_id)
    all_tags = Tag.query.all()

    if request.method == "POST":
        project.name = request.form.get("name")
        project.description = request.form.get("description")
        project.status = request.form.get("status", "active")

        # 处理标签选择
        selected_tag_ids = request.form.getlist("tags")
        project.tags = Tag.query.filter(Tag.id.in_(selected_tag_ids)).all()

        db.session.commit()
        return redirect(url_for("project_detail", project_id=project.id))

    return render_template(
        "projects/form.html", project=project, all_tags=all_tags, is_edit=True
    )


@app.route("/docs")
def doc_list():
    project_id = request.args.get("project_id")

    # 获取每个文档的最新版本
    subquery = (
        db.session.query(
            DocumentVersion.title,
            DocumentVersion.project_id,
            db.func.max(DocumentVersion.created_at).label("max_created_at"),
        )
        .group_by(DocumentVersion.title, DocumentVersion.project_id)
        .subquery()
    )

    query = DocumentVersion.query.join(
        subquery,
        db.and_(
            DocumentVersion.title == subquery.c.title,
            DocumentVersion.project_id == subquery.c.project_id,
            DocumentVersion.created_at == subquery.c.max_created_at,
        ),
    )

    if project_id:
        query = query.filter(DocumentVersion.project_id == project_id)

    documents = query.order_by(DocumentVersion.created_at.desc()).all()
    projects = Project.query.all()

    return render_template(
        "docs/list.html",
        documents=documents,
        projects=projects,
        selected_project_id=int(project_id) if project_id else None,
    )


@app.route("/docs/create", methods=["GET", "POST"])
def doc_create():
    if request.method == "POST":
        is_draft = request.headers.get("X-Draft-Save")
        version = request.form.get("version", "1.0.0")

        # 检查是否已有草稿
        draft_key = f"doc_draft_{request.remote_addr}"  # 使用IP作为临时标识
        existing_draft = None

        if is_draft:
            existing_draft = DocumentVersion.query.filter_by(
                title=request.form.get("title"),
                project_id=request.form.get("project_id"),
                is_draft=True,
                draft_key=draft_key,
            ).first()

        # 存在草稿则更新，否则创建
        if existing_draft:
            existing_draft.content = request.form.get("content")
            existing_draft.version = version
            db.session.commit()
            return jsonify(
                {"success": True, "is_update": True, "doc_id": existing_draft.id}
            )
        else:
            new_doc = DocumentVersion(
                title=request.form.get("title"),
                content=request.form.get("content"),
                project_id=request.form.get("project_id"),
                version=version,
                author_id=1,
                is_draft=bool(is_draft),
                draft_key=draft_key if is_draft else None,
                created_at=datetime.utcnow(),
            )
            db.session.add(new_doc)
            db.session.commit()

            if is_draft:
                return jsonify(
                    {"success": True, "is_update": False, "doc_id": new_doc.id}
                )
            return redirect(url_for("doc_detail", doc_id=new_doc.id))

    projects = Project.query.all()
    return render_template("docs/form.html", projects=projects, is_edit=False)


@app.route("/docs/<int:doc_id>/edit", methods=["GET", "POST"])
def doc_edit(doc_id):
    current_doc = DocumentVersion.query.get_or_404(doc_id)

    if request.method == "POST":
        is_draft = request.headers.get("X-Draft-Save")
        new_version = request.form.get("version", current_doc.version)

        # 草稿保存逻辑
        if is_draft:
            if new_version == current_doc.version:
                # 版本相同则更新当前文档
                current_doc.content = request.form.get("content")
                current_doc.title = request.form.get("title")
                db.session.commit()
                return jsonify({"success": True, "is_update": True})
            else:
                # 版本不同则创建新版本
                return create_new_version(current_doc, request, is_draft=True)

        # 正式提交逻辑
        if new_version != current_doc.version:
            return create_new_version(current_doc, request)

        # 版本相同但非草稿保存
        current_doc.content = request.form.get("content")
        current_doc.title = request.form.get("title")
        db.session.commit()
        return redirect(url_for("doc_detail", doc_id=current_doc.id))

    projects = Project.query.all()
    return render_template(
        "docs/form.html", document=current_doc, projects=projects, is_edit=True
    )


def create_new_version(base_doc, request, is_draft=False):
    """创建新版本文档的公共方法"""
    new_doc = DocumentVersion(
        title=request.form.get("title"),
        content=request.form.get("content"),
        project_id=base_doc.project_id,
        version=request.form.get("version"),
        author_id=1,
        created_at=datetime.utcnow(),
    )
    db.session.add(new_doc)
    db.session.commit()

    if is_draft:
        return jsonify({"success": True, "is_update": False, "doc_id": new_doc.id})
    return redirect(url_for("doc_detail", doc_id=new_doc.id))


@app.route("/docs/<int:doc_id>")
def doc_detail(doc_id):
    document = DocumentVersion.query.get_or_404(doc_id)
    versions = (
        DocumentVersion.query.filter(
            DocumentVersion.title == document.title,
            DocumentVersion.project_id == document.project_id,
        )
        .order_by(DocumentVersion.created_at.desc())
        .all()
    )

    # 清理原始内容的首尾空白字符
    cleaned_content = document.content.strip()

    # 配置 Markdown 渲染，启用 MathJax 兼容的数学公式
    html_content = markdown.markdown(
        cleaned_content,
        extensions=[
            "extra",  # 支持表格、定义列表等
            "codehilite",  # 代码高亮
            "pymdownx.arithmatex",  # 数学公式支持
        ],
        extension_configs={
            "pymdownx.arithmatex": {
                "generic": True  # 输出通用格式，保留原始 TeX 代码，交给 MathJax 渲染
            }
        },
        output_format="html5",
    )

    # 存储渲染后的 HTML
    document.html_content = Markup(html_content)

    return render_template("docs/detail.html", document=document, versions=versions)


@app.route("/docs/<int:doc_id>/compare", methods=["GET", "POST"])
def doc_version_compare(doc_id):
    current_doc = DocumentVersion.query.get_or_404(doc_id)
    all_versions = (
        DocumentVersion.query.filter(
            DocumentVersion.project_id == current_doc.project_id,
            DocumentVersion.title == current_doc.title,
        )
        .order_by(DocumentVersion.created_at.desc())
        .all()
    )
    print(all_versions, "all_versions")

    # 确保至少有当前版本
    if not all_versions:
        all_versions = [current_doc]

    if request.method == "POST":
        from_version_id = request.form.get("from_version", str(all_versions[-1].id))
        to_version_id = request.form.get("to_version", str(current_doc.id))

        from_version = next(
            (v for v in all_versions if v.id == int(from_version_id)), all_versions[-1]
        )
        to_version = next(
            (v for v in all_versions if v.id == int(to_version_id)), current_doc
        )
    else:
        # 默认比较：从最新版本到当前版本
        from_version = all_versions[-1]
        to_version = current_doc

    differ = difflib.HtmlDiff()
    diff_content = differ.make_table(
        from_version.content.splitlines(),
        to_version.content.splitlines(),
        fromdesc=f"Version {from_version.version}",
        todesc=f"Version {to_version.version}",
    )

    return render_template(
        "docs/compare.html",
        document=current_doc,
        all_versions=all_versions,
        from_version=from_version,
        to_version=to_version,
        diff_content=diff_content,
    )


def get_next_version(project_id, title, is_major_update=False):
    last_version = (
        DocumentVersion.query.filter(
            DocumentVersion.project_id == project_id, DocumentVersion.title == title
        )
        .order_by(DocumentVersion.created_at.desc())
        .first()
    )

    if not last_version:
        return "1.0.0"

    try:
        major, minor, patch = map(int, last_version.version.split("."))
        if is_major_update:
            return f"{major + 1}.0.0"
        else:
            return f"{major}.{minor + 1}.0"
    except:
        return last_version.version + ".1"


@app.route("/api/next-version")
def get_next_version_api():
    project_id = request.args.get("project_id")
    title = request.args.get("title")
    is_major = request.args.get("major", "false").lower() == "true"
    return jsonify({"version": get_next_version(project_id, title, is_major)})


@app.route("/api/experiments", methods=["POST"])
def create_experiment():
    data = request.get_json()
    new_exp = Experiment(
        project_id=data["project_id"],
        name=data["name"],
        description=data.get("description", ""),
        parameters=json.dumps(data.get("parameters", {})),
        metrics=json.dumps(data.get("metrics", {})),
        created_by=1,  # 假设当前用户ID为1
    )
    db.session.add(new_exp)
    db.session.commit()
    return jsonify({"message": "Experiment created", "experiment_id": new_exp.id}), 201


# 初始化数据库
with app.app_context():
    db.create_all()

# from apscheduler.schedulers.background import BackgroundScheduler


def cleanup_drafts():
    """每天清理超过7天的草稿"""
    with app.app_context():
        cutoff = datetime.utcnow() - timedelta(days=7)
        DocumentVersion.query.filter(
            DocumentVersion.is_draft == True, DocumentVersion.created_at < cutoff
        ).delete()
        db.session.commit()


# 启动定时任务
# scheduler = BackgroundScheduler()
# scheduler.add_job(cleanup_drafts, 'interval', days=1)
# scheduler.start()


# 代码管理路由
@app.route("/code")
def code_list():
    try:
        projects = db.session.execute(db.select(Project)).scalars().all()
        selected_project_id = request.args.get(
            "project_id", projects[0].id if projects else None
        )

        if selected_project_id:
            # 获取项目
            project = db.session.get(Project, selected_project_id)
            if not project:
                flash("无效的项目ID", "error")
                return redirect(url_for("code_list"))

            # 获取版本列表（按时间倒序）
            versions = (
                db.session.execute(
                    db.select(CodeVersion)
                    .filter_by(project_id=selected_project_id)
                    .order_by(CodeVersion.created_at.desc())
                )
                .scalars()
                .all()
            )

            # 获取最新版本的文件结构
            current_version = versions[0] if versions else None
            files = []
            if current_version and os.path.exists(current_version.file_path):
                for item in os.listdir(current_version.file_path):
                    item_path = os.path.join(current_version.file_path, item)
                    files.append(
                        {
                            "name": item,
                            "path": item_path,
                            "is_dir": os.path.isdir(item_path),
                            "version": current_version.version,
                        }
                    )
        else:
            project = None
            versions = []
            current_version = None
            files = []

        return render_template(
            "code/list.html",
            projects=projects,
            selected_project_id=(
                int(selected_project_id) if selected_project_id else None
            ),
            current_project=project,
            current_version=current_version,
            versions=versions,
            files=files,
        )

    except Exception as e:
        app.logger.error(f"Error in code_list: {str(e)}", exc_info=True)
        flash("加载代码列表时出错", "error")
        return redirect(url_for("code_list"))


@app.route("/code/upload", methods=["POST"])
# @login_required
def code_upload():
    try:
        project_id = request.form.get("project_id")
        if not project_id:
            flash("未选择项目", "error")
            return redirect(url_for("code_list"))

        project = db.session.get(Project, project_id)
        if not project:
            flash("无效的项目ID", "error")
            return redirect(url_for("code_list"))

        # 获取上传参数
        upload_type = request.form.get("upload_type", "file")
        version = request.form.get("version", "1.0.0")
        commit_message = request.form.get("message", "初始提交")
        files = request.files.getlist("file")

        # 验证文件
        if not files or not files[0].filename:
            flash("未选择文件", "error")
            return redirect(url_for("code_list", project_id=project_id))

        # 创建版本目录
        version_dir = os.path.join(
            app.config["UPLOAD_FOLDER"], "code", f"project_{project_id}", f"v{version}"
        )
        os.makedirs(version_dir, exist_ok=True)

        # 处理上传
        saved_files = []
        for file in files:
            if file and allowed_file(file.filename):
                # 处理路径
                if upload_type == "folder" and hasattr(file, "webkitRelativePath"):
                    rel_path = os.path.dirname(file.webkitRelativePath)
                    dest_dir = os.path.join(version_dir, rel_path)
                    os.makedirs(dest_dir, exist_ok=True)
                    filename = secure_filename(os.path.basename(file.filename))
                    filepath = os.path.join(dest_dir, filename)
                else:
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(version_dir, filename)

                # 保存文件
                file.save(filepath)
                saved_files.append(
                    {"path": filepath, "size": os.path.getsize(filepath)}
                )

        if not saved_files:
            flash("没有有效的文件被保存", "error")
            return redirect(url_for("code_list", project_id=project_id))

        # 创建版本记录
        new_version = CodeVersion(
            project_id=project_id,
            version=version,
            file_path=version_dir,
            commit_message=commit_message,
            author_id=0,  # current_user.id,
            file_count=len(saved_files),
            total_size=sum(f["size"] for f in saved_files),
        )

        db.session.add(new_version)
        db.session.commit()

        flash(f"成功上传 {len(saved_files)} 个文件到版本 {version}", "success")
        return redirect(url_for("code_list", project_id=project_id))

    except Exception as e:
        db.session.rollback()
        app.logger.error(f"上传失败: {str(e)}", exc_info=True)
        flash(f"上传失败: {str(e)}", "error")
        return redirect(
            url_for(
                "code_list", project_id=project_id if "project_id" in locals() else None
            )
        )


@app.route("/code/<int:version_id>")
def code_detail(version_id):
    try:
        version = CodeVersion.query.get_or_404(version_id)

        if not os.path.exists(version.file_path):
            flash("File not found on server", "error")
            return redirect(url_for("code_list", project_id=version.project_id))

        with open(version.file_path, "r") as f:
            file_content = f.read()

        return render_template(
            "code/detail.html", version=version, file_content=file_content
        )

    except Exception as e:
        flash(f"Error loading code version: {str(e)}", "error")
        return redirect(url_for("code_list"))


@app.route("/code/download")
def code_download():
    """下载文件"""
    file_path = request.args.get("path")
    if not file_path or not os.path.exists(file_path):
        flash("File not found", "error")
        return redirect(url_for("code_list"))

    if os.path.isdir(file_path):
        # 如果是目录，创建zip文件
        import zipfile
        from io import BytesIO

        memory_file = BytesIO()
        with zipfile.ZipFile(memory_file, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(file_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, os.path.dirname(file_path))
                    zf.write(file_path, arcname)

        memory_file.seek(0)
        filename = os.path.basename(file_path) + ".zip"
        return send_file(
            memory_file,
            mimetype="application/zip",
            as_attachment=True,
            download_name=filename,
        )
    else:
        return send_file(file_path, as_attachment=True)


def get_code_files(project):
    """获取项目代码文件结构"""
    files = []
    if project and project.code_versions:
        try:
            latest_version = project.code_versions[-1]
            code_dir = os.path.dirname(latest_version.file_path)

            for root, dirs, filenames in os.walk(code_dir):
                # 相对路径
                rel_path = os.path.relpath(root, code_dir)

                for dirname in dirs:
                    files.append(
                        {
                            "name": dirname,
                            "path": os.path.join(root, dirname),
                            "rel_path": os.path.join(rel_path, dirname),
                            "is_dir": True,
                        }
                    )

                for filename in filenames:
                    files.append(
                        {
                            "name": filename,
                            "path": os.path.join(root, filename),
                            "rel_path": os.path.join(rel_path, filename),
                            "is_dir": False,
                        }
                    )
        except Exception as e:
            print(f"Error getting code files: {e}")
            return []

    return files


@app.route("/api/code/file")
def get_file_content():
    """获取文件内容API"""
    file_path = request.args.get("path")
    if not file_path or not os.path.exists(file_path) or os.path.isdir(file_path):
        return jsonify({"error": "File not found"}), 404

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 自动检测语言
        language = "plaintext"
        if "." in file_path:
            ext = file_path.rsplit(".", 1)[1].lower()
            language = {
                "py": "python",
                "js": "javascript",
                "html": "html",
                "css": "css",
                "md": "markdown",
            }.get(ext, "plaintext")

        return jsonify(
            {
                "content": content,
                "language": language,
                "size": os.path.getsize(file_path),
            }
        )
    except UnicodeDecodeError:
        return jsonify({"error": "Binary file cannot be displayed"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.template_filter("highlight_code")
def highlight_code_filter(code, language="python"):
    """代码高亮过滤器"""
    try:
        lexer = pygments.lexers.get_lexer_by_name(language)
        formatter = HtmlFormatter(style="friendly", noclasses=True)
        return Markup(pygments.highlight(code, lexer, formatter))
    except:
        return Markup(f"<pre><code>{code}</code></pre>")


@app.route("/api/code/files")
def get_project_files():
    """获取项目文件列表API"""
    project_id = request.args.get("project_id")
    if not project_id:
        return jsonify({"error": "Project ID required"}), 400

    project = db.session.get(Project, project_id)
    if not project:
        return jsonify({"error": "Project not found"}), 404

    files = get_code_files(project)
    return jsonify(
        {
            "files": [
                {
                    "name": f["name"],
                    "path": f["path"],
                    "rel_path": f.get("rel_path", f["name"]),
                    "is_dir": f["is_dir"],
                }
                for f in files
            ]
        }
    )


@app.route("/api/code/folder")
def get_folder_contents():
    path = request.args.get("path")
    if not path or not os.path.exists(path):
        return jsonify({"error": "Folder not found"}), 404

    contents = []
    try:
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            contents.append(
                {"name": item, "path": item_path, "is_dir": os.path.isdir(item_path)}
            )
        return jsonify(contents)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 查看特定版本
@app.route("/code/version/<int:version_id>")
def code_version(version_id):
    version = db.session.get(CodeVersion, version_id)
    if not version:
        flash("版本不存在", "error")
        return redirect(url_for("code_list"))

    # 获取文件列表
    files = []
    if os.path.exists(version.file_path):
        for item in os.listdir(version.file_path):
            item_path = os.path.join(version.file_path, item)
            files.append(
                {
                    "name": item,
                    "path": item_path,
                    "is_dir": os.path.isdir(item_path),
                    "size": (
                        os.path.getsize(item_path)
                        if not os.path.isdir(item_path)
                        else 0
                    ),
                }
            )

    return render_template(
        "code/version.html",
        version=version,
        files=files,
        current_project=version.project,
    )


# 下载特定版本
@app.route("/code/version/<int:version_id>/download")
def download_version(version_id):
    version = db.session.get(CodeVersion, version_id)
    if not version:
        flash("版本不存在", "error")
        return redirect(url_for("code_list"))

    if not os.path.exists(version.file_path):
        flash("文件不存在", "error")
        return redirect(url_for("code_list", project_id=version.project_id))

    # 如果是目录则打包下载
    if os.path.isdir(version.file_path):
        from io import BytesIO
        import zipfile

        memory_file = BytesIO()
        with zipfile.ZipFile(memory_file, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(version.file_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, version.file_path)
                    zf.write(file_path, arcname)

        memory_file.seek(0)
        return send_file(
            memory_file,
            mimetype="application/zip",
            as_attachment=True,
            download_name=f"version_{version.version}.zip",
        )
    else:
        # 单文件直接下载
        return send_file(
            version.file_path,
            as_attachment=True,
            download_name=os.path.basename(version.file_path),
        )


@app.route("/api/code/files")
def get_files():
    """获取文件列表API"""
    version_id = request.args.get("version_id")
    if not version_id:
        return jsonify({"error": "version_id required"}), 400

    version = db.session.get(CodeVersion, version_id)
    if not version:
        return jsonify({"error": "Version not found"}), 404

    files = []
    if os.path.exists(version.file_path):
        for item in os.listdir(version.file_path):
            item_path = os.path.join(version.file_path, item)
            files.append(
                {
                    "name": item,
                    "path": item_path,
                    "is_dir": os.path.isdir(item_path),
                    "size": (
                        os.path.getsize(item_path)
                        if not os.path.isdir(item_path)
                        else 0
                    ),
                }
            )

    return jsonify({"files": files})


# 特征管理路由组
# ======================
# 特征管理路由
# ======================


@app.route("/features")
def feature_list():
    project_id = request.args.get("project_id")
    search_query = request.args.get("q", "").strip()

    query = Feature.query

    if project_id:
        query = query.filter_by(project_id=project_id)

    if search_query:
        query = query.filter(
            db.or_(
                Feature.name.ilike(f"%{search_query}%"),
                Feature.description.ilike(f"%{search_query}%"),
            )
        )

    features = query.order_by(Feature.updated_at.desc()).all()
    projects = Project.query.all()

    return render_template(
        "features/list.html",
        features=features,
        projects=projects,
        selected_project_id=int(project_id) if project_id else None,
        search_query=search_query,
    )


@app.route("/features/create", methods=["GET", "POST"])
def feature_create():
    """创建新特征"""
    if request.method == "POST":
        try:
            # 获取表单数据
            feature_data = {
                "name": request.form.get("name"),
                "description": request.form.get("description"),
                "data_type": request.form.get("data_type"),
                "project_id": request.form.get("project_id"),
                "sample_data": request.form.get("sample_data"),
                "schema_definition": request.form.get("schema_definition"),
            }

            # 验证数据
            if not feature_data["name"] or not feature_data["project_id"]:
                flash("Name and project are required", "error")
                return redirect(url_for("feature_create"))

            # 创建特征
            new_feature = Feature(
                name=feature_data["name"],
                short_description=request.form.get("short_description"),
                long_description=request.form.get("long_description"),
                data_type=feature_data["data_type"],
                project_id=feature_data["project_id"],
            )
            db.session.add(new_feature)
            db.session.flush()  # 获取feature_id

            # 创建初始版本
            initial_version = FeatureVersion(
                feature_id=new_feature.id,
                version="1.0.0",
                sample_data=feature_data["sample_data"],
                schema_definition=feature_data["schema_definition"],
                created_by=1,  # 替换为当前用户ID
            )
            db.session.add(initial_version)

            db.session.commit()
            flash("Feature created successfully", "success")
            return redirect(url_for("feature_detail", feature_id=new_feature.id))

        except Exception as e:
            db.session.rollback()
            flash(f"Error creating feature: {str(e)}", "error")
            return redirect(url_for("feature_create"))

    # GET请求显示表单
    projects = Project.query.all()
    data_types = ["string", "number", "boolean", "datetime", "array", "tensor", "dict"]

    return render_template(
        "features/form.html", projects=projects, data_types=data_types, is_edit=False
    )


@app.route("/features/<int:feature_id>/edit", methods=["GET", "POST"])
def feature_edit(feature_id):
    """编辑特征基本信息"""
    feature = db.session.get(Feature, feature_id)
    if not feature:
        flash("Feature not found", "error")
        return redirect(url_for("feature_list"))

    if request.method == "POST":
        try:
            feature.name = request.form.get("name")
            feature.short_description = request.form.get("short_description")
            feature.long_description = request.form.get("long_description")

            feature.data_type = request.form.get("data_type")
            feature.project_id = request.form.get("project_id")
            feature.updated_at = datetime.utcnow()

            db.session.commit()
            flash("Feature updated successfully", "success")
            return redirect(url_for("feature_detail", feature_id=feature.id))
        except Exception as e:
            db.session.rollback()
            flash(f"Error updating feature: {str(e)}", "error")
            return redirect(url_for("feature_edit", feature_id=feature_id))

    projects = Project.query.all()
    data_types = ["string", "number", "boolean", "datetime", "array", "tensor"]

    return render_template(
        "features/form.html",
        feature=feature,
        projects=projects,
        data_types=data_types,
        is_edit=True,
    )


@app.route("/features/<int:feature_id>")
def feature_detail(feature_id):
    """特征详情页"""
    feature = db.session.get(Feature, feature_id)
    if not feature:
        flash("特征不存在", "error")
        return redirect(url_for("feature_list"))

    # 获取所有版本
    versions = (
        FeatureVersion.query.filter_by(feature_id=feature_id)
        .order_by(FeatureVersion.created_at.desc())
        .all()
    )

    # 尝试解析JSON数据
    current_version = versions[0] if versions else None
    sample_data = {}
    schema = {}

    if current_version:
        try:
            sample_data = (
                json.loads(current_version.sample_data)
                if current_version.sample_data
                else {}
            )
            schema = (
                json.loads(current_version.schema_definition)
                if current_version.schema_definition
                else {}
            )
        except json.JSONDecodeError:
            pass

    return render_template(
        "features/detail.html",
        feature=feature,
        versions=versions,
        current_version=current_version,
        sample_data=sample_data,
        schema=schema,
    )


@app.route("/features/<int:feature_id>/versions/new", methods=["GET", "POST"])
def feature_version_create(feature_id):
    """创建特征新版本"""
    feature = db.session.get(Feature, feature_id)
    if not feature:
        flash("特征不存在", "error")
        return redirect(url_for("feature_list"))

    if request.method == "POST":
        try:
            # 获取上一个版本
            last_version = (
                FeatureVersion.query.filter_by(feature_id=feature_id)
                .order_by(FeatureVersion.created_at.desc())
                .first()
            )

            # 创建新版本
            version = FeatureVersion(
                feature_id=feature_id,
                version=request.form.get("version"),
                sample_data=request.form.get("sample_data"),
                schema_definition=request.form.get("schema_definition"),
                created_by=1,  # 替换为当前用户ID
                created_at=datetime.utcnow(),
            )
            db.session.add(version)

            # 更新特征更新时间
            feature.updated_at = datetime.utcnow()

            db.session.commit()
            flash("特征版本创建成功", "success")
            return redirect(url_for("feature_detail", feature_id=feature_id))
        except Exception as e:
            db.session.rollback()
            flash(f"创建版本失败: {str(e)}", "error")
            return redirect(url_for("feature_version_create", feature_id=feature_id))

    # 计算下一个版本号
    last_version = (
        FeatureVersion.query.filter_by(feature_id=feature_id)
        .order_by(FeatureVersion.created_at.desc())
        .first()
    )

    if last_version:
        try:
            major, minor, patch = map(int, last_version.version.split("."))
            next_version = f"{major}.{minor}.{patch + 1}"
        except:
            next_version = f"{last_version.version}.1"
    else:
        next_version = "1.0.0"

    return render_template(
        "features/version_form.html",
        feature=feature,
        next_version=next_version,
        last_version=last_version,
    )


@app.route("/features/<int:feature_id>/analyze")
def feature_analyze(feature_id):
    """特征分析页面"""
    feature = db.session.get(Feature, feature_id)
    if not feature:
        flash("特征不存在", "error")
        return redirect(url_for("feature_list"))

    return render_template("features/analysis.html", feature=feature)


@app.route("/api/features/<int:version_id>/analysis")
def feature_version_analysis(version_id):
    """获取特征分析数据API"""
    version = db.session.get(FeatureVersion, version_id)
    if not version:
        return jsonify({"error": "Version not found"}), 404

    try:
        # 这里应该是实际的分析逻辑，示例中使用模拟数据
        sample_data = json.loads(version.sample_data) if version.sample_data else {}

        # 模拟分析结果
        analysis_data = {
            "distribution": {
                "type": "normal" if version.data_type == "number" else "categorical",
                "labels": (
                    list(sample_data.keys())[:10]
                    if isinstance(sample_data, dict)
                    else []
                ),
                "values": (
                    list(sample_data.values())[:10]
                    if isinstance(sample_data, dict)
                    else []
                ),
            },
            "quality": {
                "missing": 0.05,  # 5%缺失值
                "unique": 0.8,  # 80%唯一值
                "valid": 0.95,  # 95%有效值
                "range": "0-1" if version.data_type == "number" else "N/A",
                "stats": {
                    "mean": 0.5 if version.data_type == "number" else None,
                    "std": 0.2 if version.data_type == "number" else None,
                    "min": 0 if version.data_type == "number" else None,
                    "25%": 0.3 if version.data_type == "number" else None,
                    "50%": 0.5 if version.data_type == "number" else None,
                    "75%": 0.7 if version.data_type == "number" else None,
                    "max": 1 if version.data_type == "number" else None,
                },
            },
        }

        return jsonify(analysis_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/features/<int:feature_id>/delete", methods=["POST"])
def feature_delete(feature_id):
    """删除特征"""
    feature = db.session.get(Feature, feature_id)
    if not feature:
        flash("特征不存在", "error")
        return redirect(url_for("feature_list"))

    try:
        # 删除相关版本
        FeatureVersion.query.filter_by(feature_id=feature_id).delete()

        # 删除特征
        db.session.delete(feature)
        db.session.commit()

        flash("特征删除成功", "success")
        return redirect(url_for("feature_list"))
    except Exception as e:
        db.session.rollback()
        flash(f"删除失败: {str(e)}", "error")
        return redirect(url_for("feature_detail", feature_id=feature_id))


@app.route("/features/version/<int:version_id>/download")
def feature_version_download(version_id):
    """下载特征版本数据"""
    version = db.session.get(FeatureVersion, version_id)
    if not version:
        flash("版本不存在", "error")
        return redirect(url_for("feature_list"))

    # 创建临时JSON文件
    import tempfile
    from io import BytesIO

    data = {
        "feature_id": version.feature_id,
        "version": version.version,
        "sample_data": json.loads(version.sample_data) if version.sample_data else {},
        "schema_definition": (
            json.loads(version.schema_definition) if version.schema_definition else {}
        ),
    }

    memory_file = BytesIO()
    memory_file.write(json.dumps(data, indent=2).encode("utf-8"))
    memory_file.seek(0)

    return send_file(
        memory_file,
        mimetype="application/json",
        as_attachment=True,
        download_name=f"feature_v{version.version}.json",
    )


# ======================
# 辅助函数
# ======================


def get_feature_stats(feature):
    """获取特征统计信息"""
    if not feature.versions:
        return None

    latest_version = feature.versions[-1]
    try:
        sample_data = (
            json.loads(latest_version.sample_data) if latest_version.sample_data else {}
        )

        stats = {
            "data_type": feature.data_type,
            "versions": len(feature.versions),
            "last_updated": feature.updated_at.strftime("%Y-%m-%d"),
        }

        if (
            feature.data_type == "number"
            and isinstance(sample_data, dict)
            and sample_data
        ):
            values = [v for v in sample_data.values() if isinstance(v, (int, float))]
            if values:
                stats.update(
                    {
                        "min": min(values),
                        "max": max(values),
                        "avg": sum(values) / len(values),
                    }
                )

        return stats
    except:
        return None


@app.route("/api/features/versions/<int:version_id>")
def get_feature_version(version_id):
    """获取特征版本详情API"""
    version = db.session.get(FeatureVersion, version_id)
    if not version:
        return jsonify({"error": "Version not found"}), 404

    return jsonify(
        {
            "id": version.id,
            "version": version.version,
            "sample_data": (
                json.loads(version.sample_data) if version.sample_data else {}
            ),
            "schema_definition": (
                json.loads(version.schema_definition)
                if version.schema_definition
                else {}
            ),
            "created_at": version.created_at.strftime("%Y-%m-%d %H:%M"),
            "author": version.author.username if version.author else "Unknown",
        }
    )


#### 用户分群
# 用户分群路由组
@app.route("/segments")
def segment_list():
    project_id = request.args.get("project_id")
    search_query = request.args.get("q", "").strip()

    query = UserSegment.query

    if project_id:
        query = query.filter_by(project_id=project_id)

    if search_query:
        query = query.filter(
            db.or_(
                UserSegment.name.ilike(f"%{search_query}%"),
                UserSegment.description.ilike(f"%{search_query}%"),
            )
        )

    segments = query.order_by(UserSegment.updated_at.desc()).all()
    projects = Project.query.all()

    return render_template(
        "segments/list.html",
        segments=segments,
        projects=projects,
        selected_project_id=int(project_id) if project_id else None,
        search_query=search_query,
    )


@app.route("/segments/<int:segment_id>")
def segment_detail(segment_id):
    segment = db.session.get(UserSegment, segment_id)
    if not segment:
        flash("Segment not found", "error")
        return redirect(url_for("segment_list"))

    # 获取分群用户数量
    user_count = len(segment.users)

    # 解析过滤规则
    try:
        filter_rules = json.loads(segment.filter_rules) if segment.filter_rules else {}
    except json.JSONDecodeError:
        filter_rules = {}

    # 获取关联的营销活动
    campaigns = segment.campaigns

    return render_template(
        "segments/detail.html",
        segment=segment,
        user_count=user_count,
        filter_rules=filter_rules,
        campaigns=campaigns,
    )


@app.route("/segments/<int:segment_id>/users")
def segment_users(segment_id):
    segment = db.session.get(UserSegment, segment_id)
    if not segment:
        flash("Segment not found", "error")
        return redirect(url_for("segment_list"))

    # 分页处理
    page = request.args.get("page", 1, type=int)
    per_page = 20

    users = (
        User.query.join(SegmentUser)
        .filter(SegmentUser.segment_id == segment_id)
        .order_by(SegmentUser.added_at.desc())
        .paginate(page=page, per_page=per_page)
    )

    return render_template("segments/users.html", segment=segment, users=users)


@app.route("/segments/<int:segment_id>/users/import", methods=["GET", "POST"])
def segment_import_users(segment_id):
    segment = db.session.get(UserSegment, segment_id)
    if not segment:
        flash("Segment not found", "error")
        return redirect(url_for("segment_list"))

    if request.method == "POST":
        try:
            # 处理文件上传
            if "file" not in request.files:
                flash("No file uploaded", "error")
                return redirect(request.url)

            file = request.files["file"]
            if file.filename == "":
                flash("No selected file", "error")
                return redirect(request.url)

            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(filepath)

                # 解析文件并导入用户
                import_count = 0
                if filename.endswith(".csv"):
                    import_count = import_users_from_csv(segment_id, filepath)
                elif filename.endswith(".json"):
                    import_count = import_users_from_json(segment_id, filepath)

                flash(f"Successfully imported {import_count} users", "success")
                return redirect(url_for("segment_users", segment_id=segment_id))

        except Exception as e:
            db.session.rollback()
            flash(f"Error importing users: {str(e)}", "error")
            return redirect(url_for("segment_import_users", segment_id=segment_id))

    return render_template("segments/import.html", segment=segment)


@app.route("/segments/<int:segment_id>/users/export")
def segment_export_users(segment_id):
    segment = db.session.get(UserSegment, segment_id)
    if not segment:
        flash("Segment not found", "error")
        return redirect(url_for("segment_list"))

    try:
        # 创建内存文件
        from io import StringIO
        import csv

        si = StringIO()
        cw = csv.writer(si)

        # 写入表头
        cw.writerow(["id", "username", "email", "role", "created_at"])

        # 写入用户数据
        for user in segment.users:
            cw.writerow(
                [
                    user.id,
                    user.username,
                    user.email,
                    user.role,
                    user.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                ]
            )

        # 创建响应
        output = make_response(si.getvalue())
        output.headers["Content-Disposition"] = (
            f"attachment; filename=segment_{segment_id}_users.csv"
        )
        output.headers["Content-type"] = "text/csv"
        return output

    except Exception as e:
        flash(f"Error exporting users: {str(e)}", "error")
        return redirect(url_for("segment_users", segment_id=segment_id))


# 营销活动路由
@app.route("/campaigns")
def campaign_list():
    status = request.args.get("status", "all")
    search_query = request.args.get("q", "").strip()

    query = Campaign.query

    if status != "all":
        query = query.filter_by(status=status)

    if search_query:
        query = query.filter(
            db.or_(
                Campaign.name.ilike(f"%{search_query}%"),
                Campaign.description.ilike(f"%{search_query}%"),
            )
        )

    campaigns = query.order_by(Campaign.created_at.desc()).all()
    return render_template(
        "campaigns/list.html",
        campaigns=campaigns,
        status=status,
        search_query=search_query,
    )


@app.route("/campaigns/create", methods=["GET", "POST"])
def campaign_create():
    form = CampaignForm()
    form.segment_id.choices = [(s.id, s.name) for s in UserSegment.query.all()]

    if request.method == "POST":
        try:
            data = request.get_json()

            # 验证CSRF
            # if not data.get('csrf_token') or data['csrf_token'] != session.get('csrf_token'):
            #    return jsonify({'success': False, 'message': '无效的CSRF令牌'}), 403

            # 创建活动
            new_campaign = Campaign(
                name=data["name"],
                description=data.get("description", ""),
                segment_id=data["segment_id"],
                strategy=data.get("strategy", "{}"),
                start_time=datetime.strptime(data["start_datetime"], "%Y-%m-%d %H:%M"),
                end_time=datetime.strptime(data["end_datetime"], "%Y-%m-%d %H:%M"),
                created_by=0,  # current_user.id,
                status="draft",
            )

            db.session.add(new_campaign)
            db.session.commit()

            return jsonify(
                {
                    "success": True,
                    "redirect_url": url_for(
                        "campaign_detail", campaign_id=new_campaign.id
                    ),
                }
            )

        except ValueError as e:
            return (
                jsonify({"success": False, "message": f"日期格式错误: {str(e)}"}),
                400,
            )
        except Exception as e:
            db.session.rollback()
            return jsonify({"success": False, "message": str(e)}), 500

    # GET请求显示表单
    return render_template("campaigns/create.html", form=form)


@app.route("/campaigns/<int:campaign_id>/edit", methods=["GET", "POST"])
def campaign_edit(campaign_id):
    campaign = Campaign.query.get_or_404(campaign_id)
    form = CampaignForm(obj=campaign)
    form.segment_id.choices = [(s.id, s.name) for s in UserSegment.query.all()]

    if request.method == "POST":
        try:
            data = request.get_json()

            # 更新活动信息
            campaign.name = data["name"]
            campaign.description = data.get("description", "")
            campaign.segment_id = data["segment_id"]
            campaign.strategy = data.get("strategy", "{}")
            campaign.start_time = datetime.strptime(
                data["start_datetime"], "%Y-%m-%d %H:%M"
            )
            campaign.end_time = datetime.strptime(
                data["end_datetime"], "%Y-%m-%d %H:%M"
            )

            db.session.commit()

            return jsonify(
                {
                    "success": True,
                    "redirect_url": url_for("campaign_detail", campaign_id=campaign.id),
                }
            )

        except Exception as e:
            db.session.rollback()
            return jsonify({"success": False, "message": str(e)}), 500

    # GET请求显示编辑表单
    return render_template("campaigns/edit.html", form=form, campaign=campaign)


@app.route("/campaigns/<int:campaign_id>")
def campaign_detail(campaign_id):
    campaign = Campaign.query.get_or_404(campaign_id)
    return render_template("campaigns/detail.html", campaign=campaign)


@app.route("/debug/templates")
def debug_templates():
    try:
        return render_template(
            "campaigns/detail.html",
            campaign={
                "name": "测试活动",
                "segment": {"name": "测试分群"},
                "start_time": datetime.now(),
                "end_time": datetime.now(),
                "strategy": '{"type": "test"}',
            },
        )
    except Exception as e:
        return str(e), 500


@app.template_filter("average")
def average_filter(sequence):
    sequence = list(sequence)
    if not sequence:
        return 0
    return sum(sequence) / len(sequence)


@app.route("/campaigns/<int:campaign_id>/feedback")
def campaign_feedback(campaign_id):
    campaign = Campaign.query.get_or_404(campaign_id)
    page = request.args.get("page", 1, type=int)
    per_page = 10

    feedbacks = (
        CampaignFeedback.query.filter_by(campaign_id=campaign_id)
        .join(User)
        .order_by(CampaignFeedback.created_at.desc())
        .paginate(page=page, per_page=per_page)
    )

    return render_template(
        "campaigns/feedback.html", campaign=campaign, feedbacks=feedbacks
    )


@app.route("/segments/create", methods=["GET", "POST"])
# @csrf.exempt  # 如果是API接口可以临时禁用CSRF
def segment_create():
    form = SegmentForm()

    # 设置项目选择
    form.project_id.choices = [(p.id, p.name) for p in Project.query.all()]
    form.project_id.choices.insert(0, (0, "-- No Project --"))

    if form.validate_on_submit():
        try:
            new_segment = UserSegment(
                name=form.name.data,
                description=form.description.data,
                project_id=form.project_id.data if form.project_id.data != 0 else None,
                filter_rules=form.filter_rules.data,
                created_by=1,  # 替换为当前用户ID
            )

            db.session.add(new_segment)
            db.session.commit()

            flash("Segment created successfully", "success")
            return redirect(url_for("segment_list"))

        except Exception as e:
            db.session.rollback()
            flash(f"Error creating segment: {str(e)}", "error")

    return render_template("segments/form.html", form=form, is_edit=False)


@app.route("/segments/<int:segment_id>/edit", methods=["GET", "POST"])
def segment_edit(segment_id):
    segment = UserSegment.query.get_or_404(segment_id)
    form = SegmentForm(obj=segment)

    # 设置项目选择
    form.project_id.choices = [(p.id, p.name) for p in Project.query.all()]
    form.project_id.choices.insert(0, (0, "-- No Project --"))

    if form.validate_on_submit():
        try:
            segment.name = form.name.data
            segment.description = form.description.data
            segment.project_id = (
                form.project_id.data if form.project_id.data != 0 else None
            )
            segment.filter_rules = form.filter_rules.data

            db.session.commit()
            flash("Segment updated successfully", "success")
            return redirect(url_for("segment_detail", segment_id=segment.id))

        except Exception as e:
            db.session.rollback()
            flash(f"Error updating segment: {str(e)}", "error")

    return render_template(
        "segments/form.html", form=form, segment=segment, is_edit=True
    )


def import_users_from_csv(segment_id, filepath):
    """从CSV文件导入用户到分群"""
    import csv

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        user_ids = [row["id"] for row in reader if "id" in row]

    return add_users_to_segment(segment_id, user_ids)


def import_users_from_json(segment_id, filepath):
    """从JSON文件导入用户到分群"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        user_ids = [user["id"] for user in data if "id" in user]

    return add_users_to_segment(segment_id, user_ids)


def add_users_to_segment(segment_id, user_ids):
    """添加用户到分群"""
    existing_users = set(
        db.session.query(SegmentUser.user_id).filter_by(segment_id=segment_id).all()
    )
    existing_users = {u[0] for u in existing_users}

    added_count = 0
    for user_id in user_ids:
        if int(user_id) not in existing_users:
            segment_user = SegmentUser(
                segment_id=segment_id, user_id=user_id, added_by=1  # 替换为当前用户ID
            )
            db.session.add(segment_user)
            added_count += 1

    db.session.commit()
    return added_count


if __name__ == "__main__":
    app.run(debug=True, port=5010)
