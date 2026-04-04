"use client";

import { useRouter } from "next/navigation";
import { ArrowLeft, ShieldCheck, AlertTriangle, Brain, Database, Lock, Globe, Mail } from "lucide-react";
import { Button } from "@/components/ui/button";

export default function PolicyPage() {
  const router = useRouter();

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-teal-50">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-sm border-b border-gray-200 sticky top-0 z-10">
        <div className="max-w-3xl mx-auto px-4 py-3 flex items-center gap-3">
          <Button variant="ghost" size="icon" onClick={() => router.back()}>
            <ArrowLeft className="h-5 w-5 text-gray-500" />
          </Button>
          <div className="flex items-center gap-2">
            <ShieldCheck className="h-5 w-5 text-blue-600" />
            <h1 className="font-semibold text-gray-900 text-sm">
              Политика конфиденциальности и Условия использования
            </h1>
          </div>
        </div>
      </header>

      <div className="max-w-3xl mx-auto px-4 py-8 space-y-8">
        {/* MVP Warning Banner */}
        <div className="bg-amber-50 border-2 border-amber-300 rounded-2xl p-6 space-y-3">
          <div className="flex items-center gap-2">
            <AlertTriangle className="h-6 w-6 text-amber-600" />
            <h2 className="text-lg font-bold text-amber-800">
              ДЕМОНСТРАЦИОННЫЙ ПРОТОТИП (MVP)
            </h2>
          </div>
          <div className="text-sm text-amber-900 space-y-2">
            <p>
              <strong>TMS (Therapist Machine Support)</strong> — это демонстрационный прототип
              (Minimum Viable Product), разработанный в учебно-исследовательских целях.
              Данный сервис <strong>НЕ является</strong> зарегистрированным медицинским изделием,
              медицинской информационной системой или средством диагностики.
            </p>
            <p>
              Все диагнозы, оценки рисков, рекомендации и иная медицинская информация,
              генерируемая сервисом, <strong>НЕ МОГУТ использоваться</strong> для принятия
              реальных медицинских решений, назначения лечения или замены консультации
              квалифицированного врача.
            </p>
          </div>
        </div>

        {/* Last updated */}
        <p className="text-xs text-gray-400">
          Последнее обновление: 19 февраля 2026 г.
        </p>

        {/* Section 1 */}
        <section className="space-y-3">
          <h2 className="text-lg font-bold text-gray-900 flex items-center gap-2">
            <span className="w-7 h-7 rounded-lg bg-blue-100 flex items-center justify-center text-sm font-bold text-blue-700">1</span>
            Общие положения
          </h2>
          <div className="text-sm text-gray-700 space-y-2 pl-9">
            <p>
              <strong>1.1.</strong> Настоящая Политика конфиденциальности описывает порядок сбора,
              использования и защиты информации при использовании веб-приложения TMS (Therapist Machine
              Support) (далее — «Сервис»).
            </p>
            <p>
              <strong>1.2.</strong> Используя Сервис, вы (далее — «Пользователь») выражаете своё
              полное согласие с условиями настоящей Политики. Если вы не согласны, вы обязаны
              немедленно прекратить использование Сервиса.
            </p>
            <p>
              <strong>1.3.</strong> Сервис является демонстрационным прототипом (MVP), разработанным
              в учебно-исследовательских целях. Он не прошёл сертификацию как медицинское изделие и
              не предназначен для использования в клинической практике.
            </p>
          </div>
        </section>

        {/* Section 2 — Medical Disclaimer */}
        <section className="space-y-3">
          <h2 className="text-lg font-bold text-gray-900 flex items-center gap-2">
            <span className="w-7 h-7 rounded-lg bg-red-100 flex items-center justify-center text-sm font-bold text-red-700">2</span>
            Медицинский отказ от ответственности (Medical Disclaimer)
          </h2>
          <div className="bg-red-50 border border-red-200 rounded-xl p-4 ml-9">
            <p className="text-sm font-bold text-red-700 mb-3">
              ВНИМАНИЕ: СЕРВИС НЕ ЯВЛЯЕТСЯ МЕДИЦИНСКОЙ ОРГАНИЗАЦИЕЙ, МЕДИЦИНСКИМ ИЗДЕЛИЕМ ИЛИ СРЕДСТВОМ ДИАГНОСТИКИ.
            </p>
            <div className="text-sm text-gray-700 space-y-2">
              <p>
                <strong>2.1.</strong> Вся информация, предоставляемая Сервисом (диагнозы, оценки рисков,
                рекомендации по специалистам и лечению), генерируется автоматически с использованием
                алгоритмов искусственного интеллекта и упрощённого математического моделирования.
                Модель обучена на ограниченном наборе из 13 заболеваний и 23 симптомов и
                <strong> не отражает полноту клинической картины</strong>.
              </p>
              <p>
                <strong>2.2.</strong> Информация носит исключительно справочно-демонстрационный характер
                и <strong>НЕ МОЖЕТ</strong> заменить профессиональную медицинскую консультацию, диагностику
                или лечение. Любые «диагнозы» являются <strong>предположительными</strong> и основаны
                на вероятностных расчётах, а не на клиническом обследовании.
              </p>
              <p>
                <strong>2.3.</strong> Разработчики Сервиса <strong>не несут ответственности</strong> за любые
                решения, принятые Пользователем на основе данных Сервиса, включая, но не ограничиваясь:
                приём лекарственных препаратов, отказ от обращения к врачу, самолечение.
              </p>
              <p>
                <strong>2.4.</strong> В случае острой боли, угрозы жизни или неотложных состояний
                Пользователь обязан <strong>немедленно</strong> обратиться в службу скорой медицинской
                помощи (<strong>103 / 112</strong>).
              </p>
              <p>
                <strong>2.5.</strong> Раздел «Портал для врачей» является демонстрацией концепции
                AI-ассистента. Клинические обоснования, генерируемые AI, предназначены для ознакомления
                и <strong>не являются</strong> клиническими рекомендациями. Врач несёт полную
                ответственность за принятые решения.
              </p>
            </div>
          </div>
        </section>

        {/* Section 3 — Data */}
        <section className="space-y-3">
          <h2 className="text-lg font-bold text-gray-900 flex items-center gap-2">
            <span className="w-7 h-7 rounded-lg bg-blue-100 flex items-center justify-center text-sm font-bold text-blue-700">3</span>
            Собираемые данные
          </h2>
          <div className="text-sm text-gray-700 space-y-2 pl-9">
            <p className="font-medium text-gray-900">
              Мы придерживаемся принципа минимизации данных (Privacy by Design).
            </p>
            <p>
              Мы <strong>НЕ собираем</strong> персональные данные (PII), позволяющие прямо
              идентифицировать личность (паспортные данные, телефон, Email, адрес).
            </p>
            <p>Мы обрабатываем следующие категории данных:</p>
            <ul className="list-disc pl-5 space-y-1">
              <li>
                <strong>3.1.</strong> Имя и город — вводятся Пользователем добровольно. Могут быть
                вымышленными. Используются для удобства идентификации врачом.
              </li>
              <li>
                <strong>3.2.</strong> Текстовые описания симптомов, введённые Пользователем.
              </li>
              <li>
                <strong>3.3.</strong> Технические данные: IP-адрес, тип устройства, тип браузера —
                для обеспечения безопасности.
              </li>
              <li>
                <strong>3.4.</strong> Код сопряжения (Pairing Code): уникальный идентификатор
                (например, LION-45), используемый для связи данных Пациента и Врача.
              </li>
              <li>
                <strong>3.5.</strong> Результаты анализа: предположительные диагнозы, оценки
                вероятности, AI-обоснования — хранятся в привязке к коду сопряжения.
              </li>
            </ul>
          </div>
        </section>

        {/* Section 4 — AI */}
        <section className="space-y-3">
          <h2 className="text-lg font-bold text-gray-900 flex items-center gap-2">
            <Brain className="h-5 w-5 text-purple-600" />
            <span>4. Использование Искусственного Интеллекта (LLM)</span>
          </h2>
          <div className="text-sm text-gray-700 space-y-2 pl-9">
            <p>
              <strong>4.1.</strong> Для анализа текстовых симптомов и генерации клинических обоснований
              Сервис использует API сторонних провайдеров (Groq Inc., США), модель Llama 3.3 70B.
            </p>
            <p>
              <strong>4.2.</strong> Пользователь соглашается с тем, что введённый им текст описания
              симптомов (без метаданных) будет передан на серверы провайдера ИИ для обработки.
            </p>
            <p>
              <strong>4.3.</strong> Мы <strong>не передаём</strong> провайдеру ИИ никаких данных,
              позволяющих идентифицировать Пользователя (Code, IP, имя и т.д. не передаются в LLM).
            </p>
            <p>
              <strong>4.4.</strong> Результаты работы AI являются вероятностными и могут содержать
              ошибки. AI-модель не имеет доступа к медицинской истории, анализам и физикальному
              обследованию пациента.
            </p>
          </div>
        </section>

        {/* Section 5 — Storage */}
        <section className="space-y-3">
          <h2 className="text-lg font-bold text-gray-900 flex items-center gap-2">
            <Database className="h-5 w-5 text-blue-600" />
            <span>5. Хранение и передача данных</span>
          </h2>
          <div className="text-sm text-gray-700 space-y-2 pl-9">
            <p>
              <strong>5.1.</strong> Данные хранятся в базе данных на сервере приложения.
            </p>
            <p>
              <strong>5.2.</strong> Пароли (PIN-коды) хранятся в хешированном виде (PBKDF2-SHA256,
              120 000 итераций) и не могут быть восстановлены.
            </p>
            <p>
              <strong>5.3.</strong> Доступ к истории симптомов имеет только:
            </p>
            <ul className="list-disc pl-5 space-y-1">
              <li>Сам Пользователь (через браузер, авторизовавшись по коду и PIN).</li>
              <li>
                Медицинский специалист, которому Пользователь <strong>добровольно</strong> сообщил
                свой Код сопряжения.
              </li>
            </ul>
            <p>
              <strong>5.4.</strong> Мы <strong>не продаём</strong> и не передаём данные третьим лицам
              в маркетинговых или коммерческих целях.
            </p>
          </div>
        </section>

        {/* Section 6 — Cookies */}
        <section className="space-y-3">
          <h2 className="text-lg font-bold text-gray-900 flex items-center gap-2">
            <Globe className="h-5 w-5 text-teal-600" />
            <span>6. Файлы Cookie и Локальное хранилище</span>
          </h2>
          <div className="text-sm text-gray-700 space-y-2 pl-9">
            <p>
              <strong>6.1.</strong> Сервис использует технологию LocalStorage браузера для хранения
              вашего Кода сопряжения, идентификатора сессии и истории чата.
            </p>
            <p>
              <strong>6.2.</strong> Удаление кэша/данных браузера может привести к потере доступа к
              локальной истории чата. Данные в базе данных сервера при этом сохраняются и доступны
              при повторном входе по коду и PIN.
            </p>
          </div>
        </section>

        {/* Section 7 — Security */}
        <section className="space-y-3">
          <h2 className="text-lg font-bold text-gray-900 flex items-center gap-2">
            <Lock className="h-5 w-5 text-gray-600" />
            <span>7. Безопасность</span>
          </h2>
          <div className="text-sm text-gray-700 space-y-2 pl-9">
            <p>
              <strong>7.1.</strong> Мы используем протокол HTTPS для шифрования данных при передаче.
            </p>
            <p>
              <strong>7.2.</strong> Несмотря на принимаемые меры, абсолютная безопасность передачи
              информации через Интернет не может быть гарантирована. Пользователь использует Сервис
              на свой страх и риск.
            </p>
            <p>
              <strong>7.3.</strong> Учитывая демонстрационный характер Сервиса, мы настоятельно
              рекомендуем <strong>не вводить</strong> реальные персональные данные (настоящее ФИО,
              реальный адрес и т.д.).
            </p>
          </div>
        </section>

        {/* Section 8 — Limitations */}
        <section className="space-y-3">
          <h2 className="text-lg font-bold text-gray-900 flex items-center gap-2">
            <AlertTriangle className="h-5 w-5 text-amber-600" />
            <span>8. Ограничения MVP-версии</span>
          </h2>
          <div className="bg-amber-50/50 border border-amber-100 rounded-xl p-4 ml-9">
            <div className="text-sm text-gray-700 space-y-2">
              <p>
                <strong>8.1.</strong> Модель ограничена <strong>13 заболеваниями</strong> и <strong>23 симптомами</strong>.
                Реальный спектр заболеваний многократно шире.
              </p>
              <p>
                <strong>8.2.</strong> Модель не учитывает: возраст, пол, хронические заболевания,
                анамнез, результаты анализов, физикальное обследование, приём лекарств и другие
                клинически значимые факторы.
              </p>
              <p>
                <strong>8.3.</strong> Алгоритм предсказания основан на весовых коэффициентах и
                не является полноценной нейронной сетью или клинической системой поддержки решений.
              </p>
              <p>
                <strong>8.4.</strong> Список врачей в портале является демонстрационным и не
                соответствует реальным медицинским специалистам.
              </p>
              <p>
                <strong>8.5.</strong> Сервис может быть недоступен, работать с ошибками или быть
                отключён без предварительного уведомления.
              </p>
            </div>
          </div>
        </section>

        {/* Section 9 — Changes */}
        <section className="space-y-3">
          <h2 className="text-lg font-bold text-gray-900 flex items-center gap-2">
            <span className="w-7 h-7 rounded-lg bg-blue-100 flex items-center justify-center text-sm font-bold text-blue-700">9</span>
            Изменения в Политике
          </h2>
          <div className="text-sm text-gray-700 pl-9">
            <p>
              <strong>9.1.</strong> Администрация оставляет за собой право вносить изменения
              в настоящую Политику. Актуальная версия всегда доступна на этой странице.
            </p>
          </div>
        </section>

        {/* Section 10 — Contacts */}
        <section className="space-y-3">
          <h2 className="text-lg font-bold text-gray-900 flex items-center gap-2">
            <Mail className="h-5 w-5 text-blue-600" />
            <span>10. Контакты</span>
          </h2>
          <div className="text-sm text-gray-700 pl-9">
            <p>По вопросам работы Сервиса и приватности: <strong>tms.support@example.com</strong></p>
          </div>
        </section>

        {/* Bottom banner */}
        <div className="bg-gray-100 rounded-xl p-4 text-center space-y-2">
          <p className="text-xs text-gray-500">
            TMS (Therapist Machine Support) — демонстрационный прототип (MVP).
          </p>
          <p className="text-xs text-gray-400">
            Все диагнозы носят предположительный характер. Не является медицинским изделием.
          </p>
        </div>

        {/* Back button */}
        <div className="text-center pb-4">
          <Button variant="outline" onClick={() => router.push("/")}>
            <ArrowLeft className="h-4 w-4 mr-2" />
            Вернуться на главную
          </Button>
        </div>
      </div>
    </div>
  );
}
