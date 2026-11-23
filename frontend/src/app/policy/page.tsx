"use client";

import { useSyncExternalStore, type ReactNode } from "react";
import { useRouter } from "next/navigation";
import { ArrowLeft, AlertTriangle, Brain, Database, Lock, Globe, Mail, type LucideIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { AppHeader } from "@/components/brand/AppHeader";
import { Logo } from "@/components/brand/Logo";
import { getLang, type Lang } from "@/lib/i18n";
import { ps } from "@/lib/strings/pages";

// Article block: number or icon, heading in the display face, body text in the reading column
function Section({ icon: Icon, title, children }: { icon?: LucideIcon; title: string; children: ReactNode }) {
  return (
    <section className="space-y-3">
      <h2 className="flex items-center gap-2.5 font-display text-xl font-semibold leading-snug text-foreground sm:text-2xl">
        {Icon && <Icon className="h-5 w-5 shrink-0 text-primary" aria-hidden="true" />}
        <span>{title}</span>
      </h2>
      <div className="space-y-3 text-[15px] leading-[1.65] text-foreground/85">{children}</div>
    </section>
  );
}

// The stored language is read without an effect. "ru" is used on the server and during hydration.
function noSubscribe() {
  return () => {};
}
function readStoredLang(): Lang {
  return getLang();
}
function serverLang(): Lang {
  return "ru";
}

export default function PolicyPage() {
  const router = useRouter();
  const lang = useSyncExternalStore(noSubscribe, readStoredLang, serverLang);

  return (
    <div className="min-h-screen bg-background text-foreground">
      <AppHeader
        logoSlot={
          <div className="flex items-center gap-1">
            <Button variant="ghost" size="icon" onClick={() => router.back()} aria-label={ps("back", lang)}>
              <ArrowLeft />
            </Button>
            <Logo />
          </div>
        }
        actions={
          <Button variant="ghost" size="sm" onClick={() => router.push("/")} className="text-muted-foreground">
            {ps("policyBack", lang)}
          </Button>
        }
      />

      <article className="mx-auto max-w-[720px] space-y-10 px-4 py-8 sm:py-12">
        <div className="space-y-2">
          <h1 className="font-display text-3xl font-semibold leading-tight text-foreground sm:text-[32px]">
            Политика конфиденциальности и Условия использования
          </h1>
          <p className="text-[13px] text-muted-foreground">Последнее обновление: 19 февраля 2026 г.</p>
        </div>

        {/* MVP Warning Banner */}
        <div className="space-y-3 rounded-xl border border-warn-border border-l-4 border-l-warn bg-warn-bg p-5 sm:p-6">
          <div className="flex items-center gap-2.5">
            <AlertTriangle className="h-5 w-5 shrink-0 text-warn" aria-hidden="true" />
            <h2 className="font-display text-xl font-semibold text-foreground">
              ДЕМОНСТРАЦИОННЫЙ ПРОТОТИП (MVP)
            </h2>
          </div>
          <div className="space-y-2 text-[15px] leading-[1.65] text-foreground/85">
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

        {/* Section 1 */}
        <Section title="1. Общие положения">
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
        </Section>

        {/* Section 2 — Medical Disclaimer */}
        <Section title="2. Медицинский отказ от ответственности (Medical Disclaimer)">
          <div className="space-y-3 rounded-xl border border-urgent-border border-l-4 border-l-urgent bg-urgent-bg p-4 sm:p-5">
            <p className="text-sm font-bold leading-relaxed text-urgent">
              ВНИМАНИЕ: СЕРВИС НЕ ЯВЛЯЕТСЯ МЕДИЦИНСКОЙ ОРГАНИЗАЦИЕЙ, МЕДИЦИНСКИМ ИЗДЕЛИЕМ ИЛИ СРЕДСТВОМ ДИАГНОСТИКИ.
            </p>
            <div className="space-y-3 text-[15px] leading-[1.65] text-foreground/85">
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
        </Section>

        {/* Section 3 — Data */}
        <Section title="3. Собираемые данные">
          <p className="font-medium text-foreground">
            Мы придерживаемся принципа минимизации данных (Privacy by Design).
          </p>
          <p>
            Мы <strong>НЕ собираем</strong> персональные данные (PII), позволяющие прямо
            идентифицировать личность (паспортные данные, телефон, Email, адрес).
          </p>
          <p>Мы обрабатываем следующие категории данных:</p>
          <ul className="list-disc space-y-1.5 pl-5 marker:text-muted-foreground">
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
        </Section>

        {/* Section 4 — AI */}
        <Section icon={Brain} title="4. Использование Искусственного Интеллекта (LLM)">
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
        </Section>

        {/* Section 5 — Storage */}
        <Section icon={Database} title="5. Хранение и передача данных">
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
          <ul className="list-disc space-y-1.5 pl-5 marker:text-muted-foreground">
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
        </Section>

        {/* Section 6 — Cookies */}
        <Section icon={Globe} title="6. Файлы Cookie и Локальное хранилище">
          <p>
            <strong>6.1.</strong> Сервис использует технологию LocalStorage браузера для хранения
            вашего Кода сопряжения, идентификатора сессии и истории чата.
          </p>
          <p>
            <strong>6.2.</strong> Удаление кэша/данных браузера может привести к потере доступа к
            локальной истории чата. Данные в базе данных сервера при этом сохраняются и доступны
            при повторном входе по коду и PIN.
          </p>
        </Section>

        {/* Section 7 — Security */}
        <Section icon={Lock} title="7. Безопасность">
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
        </Section>

        {/* Section 8 — Limitations */}
        <Section icon={AlertTriangle} title="8. Ограничения MVP-версии">
          <div className="space-y-3 rounded-xl border border-warn-border border-l-4 border-l-warn bg-warn-bg p-4 sm:p-5">
            <div className="space-y-3 text-[15px] leading-[1.65] text-foreground/85">
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
        </Section>

        {/* Section 9 — Changes */}
        <Section title="9. Изменения в Политике">
          <p>
            <strong>9.1.</strong> Администрация оставляет за собой право вносить изменения
            в настоящую Политику. Актуальная версия всегда доступна на этой странице.
          </p>
        </Section>

        {/* Section 10 — Contacts */}
        <Section icon={Mail} title="10. Контакты">
          <p>По вопросам работы Сервиса и приватности: <strong className="text-foreground">tms.support@example.com</strong></p>
        </Section>

        {/* Bottom note */}
        <div className="space-y-1 rounded-xl border border-border bg-muted p-4 text-center">
          <p className="text-[13px] text-muted-foreground">
            TMS (Therapist Machine Support) — демонстрационный прототип (MVP).
          </p>
          <p className="text-[13px] text-muted-foreground">
            Все диагнозы носят предположительный характер. Не является медицинским изделием.
          </p>
        </div>

        {/* Back button */}
        <div className="flex justify-center pb-4">
          <Button variant="outline" size="lg" onClick={() => router.push("/")}>
            <ArrowLeft />
            {ps("policyBack", lang)}
          </Button>
        </div>
      </article>
    </div>
  );
}
