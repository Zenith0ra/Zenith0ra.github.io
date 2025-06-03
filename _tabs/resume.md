---
# the default layout is 'page'
icon: fas fa-file-alt
order: 5
---

<style>
.resume-container {
  max-width: 900px;
  margin: 0 auto;
  font-family: 'Times New Roman', serif;
}

.resume-header {
  text-align: center;
  margin-bottom: 2rem;
  border-bottom: 2px solid #e9ecef;
  padding-bottom: 1.5rem;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 3rem;
  flex-wrap: wrap;
}

.header-content {
  flex: 1;
  min-width: 300px;
}

.photo-placeholder {
  width: 150px;
  height: 200px;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 4px 8px rgba(0,0,0,0.1);
  transition: transform 0.3s ease;
}

.photo-placeholder:hover {
  transform: scale(1.05);
}

.photo-placeholder img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  border-radius: 8px;
}

.resume-header h1 {
  font-size: 2.5rem;
  margin-bottom: 0.5rem;
  color: #2c3e50;
  font-weight: 300;
}

.resume-header .subtitle {
  font-size: 1.2rem;
  color: #7f8c8d;
  font-style: italic;
  margin-bottom: 1rem;
}

.contact-info {
  display: flex;
  justify-content: center;
  flex-wrap: wrap;
  gap: 2rem;
  font-size: 0.95rem;
}

.contact-info a {
  color: #3498db;
  text-decoration: none;
  transition: color 0.3s ease;
}

.contact-info a:hover {
  color: #2980b9;
}

.resume-section {
  margin-bottom: 2.5rem;
}

.section-title {
  font-size: 1.4rem;
  color: #2c3e50;
  border-bottom: 1px solid #bdc3c7;
  padding-bottom: 0.5rem;
  margin-bottom: 1.5rem;
  font-weight: 400;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.education-item, .experience-item, .project-item, .service-item, .volunteer-item, .award-item {
  margin-bottom: 1.5rem;
  padding-left: 1rem;
  border-left: 3px solid #ecf0f1;
  transition: border-color 0.3s ease;
}

.education-item:hover, .experience-item:hover, .project-item:hover, .service-item:hover, .volunteer-item:hover, .award-item:hover {
  border-left-color: #3498db;
}

.item-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 0.5rem;
  flex-wrap: wrap;
}

.item-title {
  font-weight: 600;
  color: #2c3e50;
  font-size: 1.1rem;
}

.item-subtitle {
  color: #7f8c8d;
  font-style: italic;
  margin-top: 0.2rem;
}

.item-date {
  color: #95a5a6;
  font-size: 0.9rem;
  font-weight: 500;
}

.item-description {
  color: #5d6d7e;
  line-height: 1.6;
  margin-top: 0.5rem;
}

.skills-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1.5rem;
}

.skill-category {
  background: #f8f9fa;
  padding: 1.5rem;
  border-radius: 8px;
  border: 1px solid #e9ecef;
  transition: box-shadow 0.3s ease;
}

.skill-category:hover {
  box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

.skill-category h4 {
  color: #2c3e50;
  margin-bottom: 1rem;
  font-size: 1.1rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.skill-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.skill-tag {
  background: #3498db;
  color: white;
  padding: 0.3rem 0.8rem;
  border-radius: 15px;
  font-size: 0.85rem;
  font-weight: 500;
}

.achievements-list, .volunteer-list, .award-list {
  list-style: none;
  padding: 0;
}

.achievements-list li, .volunteer-list li, .award-list li {
  padding: 0.8rem 0;
  border-bottom: 1px solid #ecf0f1;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
}

.achievements-list li:last-child, .volunteer-list li:last-child, .award-list li:last-child {
  border-bottom: none;
}

.achievement-icon, .volunteer-icon, .award-icon {
  color: #f39c12;
  font-size: 1.2rem;
  flex-shrink: 0;
}

.achievement-content, .volunteer-content, .award-content {
  display: flex;
  align-items: center;
  gap: 1rem;
  flex: 1;
}

.achievement-text, .volunteer-text, .award-text {
  flex: 1;
  color: #5d6d7e;
  line-height: 1.5;
}

.volunteer-date, .award-date {
  color: #95a5a6;
  font-size: 0.85rem;
  font-weight: 500;
  flex-shrink: 0;
}

.interests-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
}

.interest-item {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 1rem;
  border-radius: 8px;
  text-align: center;
  transition: transform 0.3s ease;
}

.interest-item:hover {
  transform: translateY(-2px);
}

.interest-item i {
  font-size: 1.5rem;
  margin-bottom: 0.5rem;
  display: block;
}

@media (max-width: 768px) {
  .resume-header {
    flex-direction: column;
    gap: 1.5rem;
  }
  
  .resume-header h1 {
    font-size: 2rem;
  }
  
  .contact-info {
    flex-direction: column;
    gap: 0.5rem;
  }
  
  .item-header {
    flex-direction: column;
    align-items: flex-start;
  }
  
  .skills-grid {
    grid-template-columns: 1fr;
  }
  
  .achievements-list li, .volunteer-list li, .award-list li {
    flex-direction: column;
    align-items: flex-start;
  }
  
  .achievement-content, .volunteer-content, .award-content {
    width: 100%;
  }
  
  .volunteer-date, .award-date {
    align-self: flex-end;
  }
}
</style>

<div class="resume-container">
  <header class="resume-header">
    <div class="header-content">
      <h1>侯林之 (Linzhi Hou)</h1>
      <p class="subtitle">Computer Science & Technology Student | Tsinghua University</p>
      <div class="contact-info">
        <span><i class="fas fa-envelope"></i> <a href="mailto:hlz23@mails.tsinghua.edu.cn">hlz23@mails.tsinghua.edu.cn</a></span>
        <span><i class="fab fa-github"></i> <a href="https://github.com/Zenith0ra" target="_blank">Zenith0ra</a></span>
        <span><i class="fas fa-globe"></i> <a href="https://houlinzhi.com" target="_blank">houlinzhi.com</a></span>
        <span><i class="fas fa-map-marker-alt"></i> Beijing, China</span>
      </div>
    </div>
    <div class="photo-placeholder">
      <img src="/assets/img/me.jpg" alt="侯林之个人照片" />
    </div>
  </header>

  <section class="resume-section">
    <h2 class="section-title">
      <i class="fas fa-graduation-cap"></i>
      教育背景 Education
    </h2>
    <div class="education-item">
      <div class="item-header">
        <div>
          <div class="item-title">清华大学 (Tsinghua University)</div>
          <div class="item-subtitle">计算机科学与技术专业 | Computer Science and Technology</div>
        </div>
        <div class="item-date">2023 - 至今</div>
      </div>
      <div class="item-description">
        现为大二学生，专注于计算机科学基础理论学习与实践应用，涉及算法设计、数据结构、软件工程等核心课程。积极参与学术活动和技术项目，致力于在计算机科学领域持续深耕。
      </div>
    </div>
  </section>

  <section class="resume-section">
    <h2 class="section-title">
      <i class="fas fa-code"></i>
      技能专长 Technical Skills
    </h2>
    <div class="skills-grid">
      <div class="skill-category">
        <h4><i class="fas fa-laptop-code"></i> 编程语言</h4>
        <div class="skill-tags">
          <span class="skill-tag">C/C++</span>
          <span class="skill-tag">Python</span>
          <span class="skill-tag">Rust</span>
          <span class="skill-tag">HTML/CSS</span>
        </div>
      </div>
      <div class="skill-category">
        <h4><i class="fas fa-tools"></i> 开发工具</h4>
        <div class="skill-tags">
          <span class="skill-tag">Git</span>
          <span class="skill-tag">VS Code</span>
          <span class="skill-tag">Linux</span>
          <span class="skill-tag">WSL</span>
          <span class="skill-tag">Docker</span>
        </div>
      </div>
      <div class="skill-category">
        <h4><i class="fas fa-database"></i> 技术框架</h4>
        <div class="skill-tags">
          <span class="skill-tag">React</span>
          <span class="skill-tag">Django</span>
          <span class="skill-tag">Node.js</span>
        </div>
      </div>
      <div class="skill-category">
        <h4><i class="fas fa-brain"></i> 专业方向</h4>
        <div class="skill-tags">
          <span class="skill-tag">算法设计</span>
          <span class="skill-tag">人工智能</span>
          <span class="skill-tag">Web开发</span>
        </div>
      </div>
    </div>
  </section>

  <section class="resume-section">
    <h2 class="section-title">
      <i class="fas fa-users"></i>
      社工经历 Leadership Experience
    </h2>
    <div class="service-item">
      <div class="item-header">
        <div>
          <div class="item-title">计33班班长</div>
          <div class="item-subtitle">清华大学计算机系</div>
        </div>
        <div class="item-date">2024.9 - 至今</div>
      </div>
      <div class="item-description">
        担任计33班班长，负责班级日常管理、学习生活协调、同学关系维护等工作。带领班级获得甲级团支部荣誉，组织各类班级活动，提升班级凝聚力。
      </div>
    </div>
    <div class="service-item">
      <div class="item-header">
        <div>
          <div class="item-title">计33班组织委员</div>
          <div class="item-subtitle">清华大学计算机系</div>
        </div>
        <div class="item-date">2023.9 - 2024.8</div>
      </div>
      <div class="item-description">
        担任计33班组织委员，协助班级各项活动的组织与实施，负责班级团建活动策划，协调同学间的学习交流。班级在此期间获得甲级团支部荣誉。
      </div>
    </div>
  </section>

  <section class="resume-section">
    <h2 class="section-title">
      <i class="fas fa-heart"></i>
      志愿服务 Volunteer Service
    </h2>
    <ul class="volunteer-list">
      <li>
        <div class="volunteer-content">
          <i class="fas fa-hands-helping volunteer-icon"></i>
          <div class="volunteer-text">
            <strong>2025校友返校日志愿服务</strong>
          </div>
        </div>
        <div class="volunteer-date">2025-05-18</div>
      </li>
      <li>
        <div class="volunteer-content">
          <i class="fas fa-running volunteer-icon"></i>
          <div class="volunteer-text">
            <strong>计算机科学与技术系第二十一届"钟士模杯"田径运动会志愿者</strong>
          </div>
        </div>
        <div class="volunteer-date">2025-05-18</div>
      </li>
      <li>
        <div class="volunteer-content">
          <i class="fas fa-microphone volunteer-icon"></i>
          <div class="volunteer-text">
            <strong>2024-2025春季学期清华大学日常讲解活动</strong>
          </div>
        </div>
        <div class="volunteer-date">2025-05-16</div>
      </li>
      <li>
        <div class="volunteer-content">
          <i class="fas fa-graduation-cap volunteer-icon"></i>
          <div class="volunteer-text">
            <strong>清华大学第23届"情系母校"志愿活动</strong>
          </div>
        </div>
        <div class="volunteer-date">2025-03-29</div>
      </li>
      <li>
        <div class="volunteer-content">
          <i class="fas fa-tools volunteer-icon"></i>
          <div class="volunteer-text">
            <strong>第十期"清"年爱劳动</strong>
          </div>
        </div>
        <div class="volunteer-date">2025-03-18</div>
      </li>
      <li>
        <div class="volunteer-content">
          <i class="fas fa-user-friends volunteer-icon"></i>
          <div class="volunteer-text">
            <strong>2024秋季学期Program Buddy活动</strong>
          </div>
        </div>
        <div class="volunteer-date">2025-02-19</div>
      </li>
      <li>
        <div class="volunteer-content">
          <i class="fas fa-envelope volunteer-icon"></i>
          <div class="volunteer-text">
            <strong>2024秋季学期Inspire启志书信活动</strong>
          </div>
        </div>
        <div class="volunteer-date">2025-02-17</div>
      </li>
      <li>
        <div class="volunteer-content">
          <i class="fas fa-calendar volunteer-icon"></i>
          <div class="volunteer-text">
            <strong>计算机系校庆校友纪念活动现场服务</strong>
          </div>
        </div>
        <div class="volunteer-date">2025-01-04</div>
      </li>
      <li>
        <div class="volunteer-content">
          <i class="fas fa-school volunteer-icon"></i>
          <div class="volunteer-text">
            <strong>2024年高考招生志愿者</strong>
          </div>
        </div>
        <div class="volunteer-date">2024-09-08</div>
      </li>
      <li>
        <div class="volunteer-content">
          <i class="fas fa-medal volunteer-icon"></i>
          <div class="volunteer-text">
            <strong>清华大学2024年"马约翰杯"田径运动会志愿项目</strong>
          </div>
        </div>
        <div class="volunteer-date">2024-05-03</div>
      </li>
      <li>
        <div class="volunteer-content">
          <i class="fas fa-flag volunteer-icon"></i>
          <div class="volunteer-text">
            <strong>清华大学计算机系2024钟士模杯运动会</strong>
          </div>
        </div>
        <div class="volunteer-date">2024-04-12</div>
      </li>
      <li>
        <div class="volunteer-content">
          <i class="fas fa-home volunteer-icon"></i>
          <div class="volunteer-text">
            <strong>清华大学第22届"情系母校"志愿活动</strong>
          </div>
        </div>
        <div class="volunteer-date">2024-03-28</div>
      </li>
      <li>
        <div class="volunteer-content">
          <i class="fas fa-utensils volunteer-icon"></i>
          <div class="volunteer-text">
            <strong>"清"年爱劳动</strong>
          </div>
        </div>
        <div class="volunteer-date">2024-01-15</div>
      </li>
      <li>
        <div class="volunteer-content">
          <i class="fas fa-handshake volunteer-icon"></i>
          <div class="volunteer-text">
            <strong>清华大学计算机系2023年迎新</strong>
          </div>
        </div>
        <div class="volunteer-date">2023-09-14</div>
      </li>
    </ul>
  </section>

  <section class="resume-section">
    <h2 class="section-title">
      <i class="fas fa-project-diagram"></i>
      项目经历 Projects
    </h2>
    <div class="project-item">
      <div class="item-header">
        <div>
          <div class="item-title">个人技术博客网站</div>
          <div class="item-subtitle">Jekyll + GitHub Pages | 前端开发</div>
        </div>
        <div class="item-date">2024</div>
      </div>
      <div class="item-description">
        使用Jekyll静态网站生成器搭建个人技术博客，采用Chirpy主题并进行深度定制。集成了评论系统、搜索功能、PWA支持等特性，展示技术文章和学习心得。网站具备响应式设计，支持深色/浅色主题切换。
      </div>
    </div>
    <div class="project-item">
      <div class="item-header">
        <div>
          <div class="item-title">算法学习与实践</div>
          <div class="item-subtitle">数据结构与算法 | 编程练习</div>
        </div>
        <div class="item-date">2023 - 至今</div>
      </div>
      <div class="item-description">
        持续学习和实践经典算法与数据结构，在LeetCode等平台上解决编程问题。涵盖排序算法、搜索算法、动态规划、图论等主题，不断提升算法思维和编程能力。
      </div>
    </div>
    <div class="project-item">
      <div class="item-header">
        <div>
          <div class="item-title">开源项目贡献</div>
          <div class="item-subtitle">GitHub | 社区参与</div>
        </div>
        <div class="item-date">2024 - 至今</div>
      </div>
      <div class="item-description">
        积极参与开源社区，通过GitHub贡献代码和文档。学习协作开发流程，提升代码质量意识，培养团队合作精神和技术交流能力。
      </div>
    </div>
  </section>

  <section class="resume-section">
    <h2 class="section-title">
      <i class="fas fa-trophy"></i>
      荣誉奖项 Awards & Achievements
    </h2>
    <ul class="award-list">
      <li>
        <div class="award-content">
          <i class="fas fa-award award-icon"></i>
          <div class="award-text">
            <strong>志愿公益优秀奖学金</strong> - 清华大学
          </div>
        </div>
        <div class="award-date">2024年</div>
      </li>
      <li>
        <div class="award-content">
          <i class="fas fa-star award-icon"></i>
          <div class="award-text">
            <strong>新生优秀奖学金</strong> - 清华大学
          </div>
        </div>
        <div class="award-date">2023年</div>
      </li>
      <li>
        <div class="award-content">
          <i class="fas fa-medal award-icon"></i>
          <div class="award-text">
            <strong>河南省三好学生</strong> - 河南省教育厅
          </div>
        </div>
        <div class="award-date">2023年</div>
      </li>
    </ul>
  </section>

  <section class="resume-section">
    <h2 class="section-title">
      <i class="fas fa-heart"></i>
      兴趣爱好 Interests
    </h2>
    <div class="interests-grid">
      <div class="interest-item">
        <i class="fas fa-laptop-code"></i>
        <div>编程与技术</div>
      </div>
      <div class="interest-item">
        <i class="fas fa-robot"></i>
        <div>人工智能</div>
      </div>
      <div class="interest-item">
        <i class="fab fa-github"></i>
        <div>开源项目</div>
      </div>
    </div>
  </section>

  <section class="resume-section">
    <h2 class="section-title">
      <i class="fas fa-quote-left"></i>
      个人格言 Personal Motto
    </h2>
    <blockquote style="text-align: center; font-style: italic; color: #7f8c8d; font-size: 1.1rem; margin: 2rem 0; padding: 1.5rem; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #3498db;">
      "Stay hungry, stay foolish. 永远保持对知识的渴望和对未知的好奇。"
    </blockquote>
  </section>
</div> 